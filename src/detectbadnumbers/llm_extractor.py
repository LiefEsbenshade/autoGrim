"""Module for extracting statistical data from scientific papers using LLMs."""

import os
from typing import Dict, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
import json
import re
import logging

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "WARNING"),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def configure_llm() -> genai.GenerativeModel:
    """
    Configure and initialize the Gemini LLM.
    
    Returns:
        The configured Gemini model
        
    Raises:
        ValueError: If GOOGLE_API_KEY environment variable is not set
    """
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    logger.debug(f"Loaded API key: {api_key[:5]}...")  # Only log first 5 chars for security
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    
    # Initialize and return the model
    return genai.GenerativeModel('gemini-1.5-pro-001')

class PaperAnalyzer:
    def __init__(self, model, cache_dir: str = None, use_cache: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            model: The LLM model to use
            cache_dir: Directory to store cached results
            use_cache: Whether to use cached results if available
        """
        self.model = model
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cache')
        self.use_cache = use_cache
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {self.cache_dir}")
        
        # Assume API is available since it was checked in cli.py
        self.api_available = True
        
        # System prompt that guides the model's behavior
        self.system_prompt = """You are a statistical data extractor. Your task is to analyze scientific papers and extract statistical data in a structured format.

Please extract the following information:
1. Sample size (N or n) from the Methods section
2. Statistical data from tables and text, including:
   - Means and standard deviations
   - Test statistics (t, F, etc.)
   - Degrees of freedom
   - p-values
   - Effect sizes (Cohen's d, etc.)

Format your response as a JSON object with the following structure:
{
  "sample_size": number or null,
  "tables": [
    {
      "table_number": number,
      "title": "string",
      "data": [
        {
          "row_label": "string",
          "mean": number,
          "sd": number,
          "n": number (if available)
        }
      ]
    }
  ],
  "text_statistics": [
    {
      "context": "string describing what was compared",
      "conditions": [
        {
          "name": "string",
          "mean": number,
          "sd": number
        }
      ],
      "test_type": "string (t-test, F-test, etc.)",
      "degrees_of_freedom": "string",
      "test_statistic": number,
      "p_value": "string",
      "effect_size": {
        "type": "string",
        "value": number
      }
    }
  ]
}

Important:
1. Extract ALL statistical comparisons, not just significant ones
2. Include descriptive context for each comparison
3. Convert all statistics to numbers (except p-values and df which may be strings)
4. If a value is not available, omit that field rather than using null
5. Ensure the JSON is properly formatted and complete
6. Do not include any text outside the JSON object"""

    def _get_cache_path(self, pdf_path: str) -> str:
        """Get the cache file path for a given PDF."""
        pdf_name = os.path.basename(pdf_path)
        cache_name = f"{pdf_name}.json"
        return os.path.join(self.cache_dir, cache_name)

    def _load_from_cache(self, pdf_path: str) -> Optional[Dict]:
        """Try to load results from cache."""
        cache_path = self._get_cache_path(pdf_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    logging.info(f"Loading cached results from {cache_path}")
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load cache from {cache_path}: {str(e)}")
        return None

    def _save_to_cache(self, pdf_path: str, results: Dict):
        """Save results to cache."""
        cache_path = self._get_cache_path(pdf_path)
        try:
            with open(cache_path, 'w') as f:
                logging.info(f"Saving results to cache: {cache_path}")
                json.dump(results, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save cache to {cache_path}: {str(e)}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file, focusing on actual content and skipping metadata."""
        logger.info(f"Starting text extraction from {pdf_path}")
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            total_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {total_pages} pages")
            
            for i, page in enumerate(pdf_reader.pages, 1):
                logger.debug(f"Processing page {i}/{total_pages}")
                page_text = page.extract_text()
                
                # Skip pages that are mostly metadata or binary content
                if any(marker in page_text for marker in ['<?xml', '<rdf:', 'xmp.iid:', '%PDF', 'stream', 'endstream']):
                    logger.debug(f"Skipping page {i} - contains metadata markers")
                    continue
                
                # Skip pages that are mostly numbers or special characters
                if len(re.sub(r'[^a-zA-Z\s]', '', page_text)) < 50:
                    logger.debug(f"Skipping page {i} - insufficient text content")
                    continue
                
                text += page_text + "\n"
                logger.debug(f"Added content from page {i} (length: {len(page_text)} chars)")
                logger.debug(f"First 500 chars of page {i}: {page_text[:500]}")
            
            # Clean up the text
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
            text = text.strip()
            logger.info(f"Extracted {len(text)} characters of text")
            logger.debug(f"First 1000 chars of extracted text: {text[:1000]}")
            return text

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks that fit within model context window."""
        max_chunk_length = 30000  # Approximate token limit for Gemini
        chunks = []
        
        # Split text into chunks
        for i in range(0, len(text), max_chunk_length):
            chunk = text[i:i + max_chunk_length]
            chunks.append(chunk)
        
        return chunks

    def analyze_paper(self, pdf_path: str) -> Dict:
        """
        Analyze a paper using LLM.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing analysis results
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Check cache first if enabled
        cache_path = self._get_cache_path(pdf_path)
        if self.use_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached results from {cache_path}")
            return self._load_from_cache(pdf_path)
        
        # Split text into chunks if needed
        chunks = self._split_text(text)
        
        # Process each chunk
        results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            result = self._process_chunk(chunk)
            if result:
                results.append(result)
        
        # Combine results
        combined_result = self._combine_results(results)
        combined_result['filename'] = os.path.basename(pdf_path)
        
        # Cache the results if enabled
        if self.use_cache:
            logger.info(f"Saving results to cache: {cache_path}")
            self._save_to_cache(pdf_path, combined_result)
        
        return combined_result

    def _combine_results(self, results: List[Dict]) -> Dict:
        """Combine results from multiple chunks into a single result."""
        logging.info(f"Combining results from {len(results)} chunks")
        
        # Initialize combined results
        combined = {
            "sample_size": None,
            "tables": [],
            "text_statistics": []
        }
        
        # Track the first non-None sample size
        for result in results:
            if not isinstance(result, dict):
                logging.warning(f"Skipping invalid result (not a dict): {result}")
                continue
            
            # Handle sample size
            if result.get("sample_size"):
                combined["sample_size"] = result["sample_size"]
                break
        
        # Combine tables and text statistics
        for result in results:
            if not isinstance(result, dict):
                continue
            
            # Add tables if present
            if "tables" in result and isinstance(result["tables"], list):
                combined["tables"].extend(result["tables"])
            
            # Add text statistics if present
            if "text_statistics" in result and isinstance(result["text_statistics"], list):
                combined["text_statistics"].extend(result["text_statistics"])
        
        logging.info(f"Selected sample size: {combined['sample_size']}")
        logging.info(f"Combined {len(combined['tables'])} tables and {len(combined['text_statistics'])} text statistics from all chunks")
        
        return combined

    def process_papers_directory(self, directory: str) -> List[Dict]:
        """Process all PDF papers in a directory."""
        logger.info(f"Processing papers in directory: {directory}")
        results = []
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for filename in pdf_files:
            pdf_path = os.path.join(directory, filename)
            logger.info(f"Processing file: {filename}")
            try:
                # Try to load from cache first
                cached_result = self._load_from_cache(pdf_path)
                if cached_result is not None:
                    cached_result["filename"] = filename
                    results.append(cached_result)
                    logger.info(f"Used cached results for {filename}")
                    continue

                # If not in cache and API is not available, skip processing
                if not self.api_available:
                    logger.warning(f"Skipping {filename} - Gemini API not available")
                    continue

                # If not in cache, process the PDF
                result = self.analyze_paper(pdf_path)
                result["filename"] = filename
                results.append(result)
                
                logger.info(f"Successfully processed {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
        
        logger.info(f"Completed processing {len(results)} papers")
        return results

    def _process_chunk(self, chunk: str) -> Optional[Dict]:
        """Process a chunk of text using the LLM."""
        if not self.api_available:
            logger.warning("Skipping chunk processing - Gemini API not available")
            return None
            
        # Generate prompt for this chunk
        prompt = self.system_prompt + "\n\nText to analyze:\n" + chunk
        
        # Send request to Gemini API
        logger.debug("Sending request to Gemini API")
        try:
            response = self.model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    candidate_count=1,
                    stop_sequences=None,
                    max_output_tokens=8192  # Increased from 2048
                )
            )
            
            # Log raw response for debugging
            logger.debug(f"Raw API response:\n{response.text}")
            
            try:
                # Clean up response text
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                # Try to find complete JSON by looking for matching braces
                brace_count = 0
                for i, char in enumerate(response_text):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            response_text = response_text[:i+1]
                            break
                
                result = json.loads(response_text)
                logger.info(f"Successfully parsed JSON response")
                logger.debug(f"Found sample size: {result.get('sample_size')}")
                logger.debug(f"Found {len(result.get('tables', []))} tables")
                logger.debug(f"Found {len(result.get('text_statistics', []))} text statistics")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.debug(f"Raw response:\n{response.text}")
                # Try to salvage partial results if possible
                if '"text_statistics":' in response.text:
                    try:
                        # Extract just the text_statistics array
                        stats_start = response.text.index('"text_statistics":') + len('"text_statistics":')
                        stats_text = response.text[stats_start:].strip()
                        if stats_text.startswith('['):
                            # Find matching closing bracket
                            bracket_count = 1
                            for i, char in enumerate(stats_text[1:], 1):
                                if char == '[':
                                    bracket_count += 1
                                elif char == ']':
                                    bracket_count -= 1
                                    if bracket_count == 0:
                                        stats_text = stats_text[:i+1]
                                        break
                            partial_result = {
                                "sample_size": None,
                                "tables": [],
                                "text_statistics": json.loads(stats_text)
                            }
                            logger.info(f"Successfully parsed partial result with {len(partial_result['text_statistics'])} text statistics")
                            return partial_result
                    except Exception as e2:
                        logger.error(f"Failed to salvage partial results: {str(e2)}")
            
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            return None