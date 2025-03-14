"""Module for extracting statistical data from scientific papers using LLMs."""

import os
from typing import Dict, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
import json
import re
import logging
from pathlib import Path

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
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    # Log securely (only first few chars)
    if api_key:
        logger.debug(f"Using API key starting with: {api_key[:5]}...")
    
    # Configure the model
    genai.configure(api_key=api_key)
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
        self.cache_dir = cache_dir or "cache"
        self.use_cache = use_cache
        
        # Create cache directory if it doesn't exist
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Assume API is available since it was checked in cli.py
        self.api_available = True
        
        # System prompt that guides the model's behavior
        self.system_prompt = """You are a statistical data extractor. Your task is to analyze scientific papers and extract statistical data in a structured format.

Please extract ALL statistical data from tables and text, including:
   - Sample sizes (N or n) for each analysis (this is critical - each test should have its own sample size)
   - Means and standard deviations
   - Test statistics (t, F, etc.) with their EXACT values and degrees of freedom
   - p-values EXACTLY as reported (e.g., "p < .05", "p = .032", etc.)
   - Effect sizes (Cohen's d, η2, etc.) with EXACT values
   - Any other reported statistical values

Format your response as a JSON object with the following structure:
{
  "tables": [
    {
      "table_number": number,
      "title": "string",
      "data": [
        {
          "row_label": "string",
          "mean": number,
          "sd": number,
          "n": number
        }
      ]
    }
  ],
  "text_statistics": [
    {
      "context": "string describing what was compared",
      "sample_size": number,  # Sample size for this specific analysis
      "conditions": [
        {
          "name": "string",
          "mean": number,
          "sd": number,
          "n": number  # Sample size for this condition if different from overall
        }
      ],
      "test_type": "string (t-test, F-test, etc.)",
      "reported_statistics": {  # All statistical values reported for this test
          "test_statistic": "string",  # e.g., "t(28) = 2.14" - preserve EXACT format
          "p_value": "string",  # e.g., "p < .05" or "p = .032" - preserve EXACT format
          "effect_size": "string",  # e.g., "Cohen's d = 0.78" - preserve EXACT format
          "means": {  # If means are reported
              "group1": "string",  # e.g., "M = 3.45, SD = 0.67"
              "group2": "string"   # e.g., "M = 2.98, SD = 0.71"
          },
          "additional_stats": {}  # Any other statistics reported
      }
    }
  ]
}

Critical Requirements:
1. Extract and preserve the EXACT format of ALL reported statistics
2. Include the complete test statistic with degrees of freedom (e.g., "t(28) = 2.14")
3. Keep p-values in their original format (e.g., "p < .05", "p = .032")
4. Include means and standard deviations when reported
5. Each statistical test MUST have its own sample size
6. Copy values EXACTLY as they appear - do not round or reformat
7. Include ALL statistical information - even if it seems redundant
"""

    def _get_cache_path(self, pdf_path: str) -> str:
        """Get the cache file path for a given PDF."""
        pdf_name = Path(pdf_path).name
        cache_name = f"{pdf_name}.json"
        return os.path.join(self.cache_dir, cache_name)

    def _load_from_cache(self, pdf_path: str) -> Optional[Dict]:
        """Try to load results from cache."""
        if not self.use_cache:
            return None
            
        cache_path = self._get_cache_path(pdf_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    logging.info(f"Loading cached results from {cache_path}")
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in cache file: {cache_path}")
                return None
        return None

    def _save_to_cache(self, pdf_path: str, results: Dict):
        """Save results to cache."""
        if not self.use_cache:
            return
            
        cache_path = self._get_cache_path(pdf_path)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                logging.info(f"Saving results to cache: {cache_path}")
                json.dump(results, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_path}: {str(e)}")

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
            "tables": [],
            "text_statistics": []
        }
        
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
                
                # Ensure all required fields are present with default values
                if 'text_statistics' in result:
                    for stat in result['text_statistics']:
                        # Ensure basic fields exist
                        stat['test_type'] = stat.get('test_type', 'Unknown')
                        stat['context'] = stat.get('context', '')
                        stat['sample_size'] = stat.get('sample_size')
                        
                        # Ensure reported_statistics exists and has all fields
                        if 'reported_statistics' not in stat:
                            stat['reported_statistics'] = {}
                        stats = stat['reported_statistics']
                        stats['test_statistic'] = stats.get('test_statistic', '')
                        stats['p_value'] = stats.get('p_value', '')
                        stats['effect_size'] = stats.get('effect_size', '')
                        stats['means'] = stats.get('means', {})
                        stats['additional_stats'] = stats.get('additional_stats', {})
                        
                        # Ensure conditions exist
                        if 'conditions' not in stat:
                            stat['conditions'] = []
                
                logger.info(f"Successfully parsed JSON response")
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
                                "tables": [],
                                "text_statistics": json.loads(stats_text)
                            }
                            # Ensure fields exist in partial results
                            for stat in partial_result['text_statistics']:
                                stat['test_type'] = stat.get('test_type', 'Unknown')
                                stat['context'] = stat.get('context', '')
                                stat['sample_size'] = stat.get('sample_size')
                                if 'reported_statistics' not in stat:
                                    stat['reported_statistics'] = {}
                                stats = stat['reported_statistics']
                                stats['test_statistic'] = stats.get('test_statistic', '')
                                stats['p_value'] = stats.get('p_value', '')
                                stats['effect_size'] = stats.get('effect_size', '')
                                stats['means'] = stats.get('means', {})
                                stats['additional_stats'] = stats.get('additional_stats', {})
                                if 'conditions' not in stat:
                                    stat['conditions'] = []
                            logger.info(f"Successfully parsed partial result with {len(partial_result['text_statistics'])} text statistics")
                            return partial_result
                    except Exception as e2:
                        logger.error(f"Failed to salvage partial results: {str(e2)}")
            
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            return None