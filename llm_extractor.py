import os
from typing import Dict, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
import json

load_dotenv()

class PaperAnalyzer:
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        self.system_prompt = """You are a scientific paper analyzer specialized in extracting statistical data.
        Your task is to identify and extract:
        1. Sample sizes
        2. Percentages
        3. Means and standard deviations
        4. Data ranges (min and max values)
        
        Format your response as a JSON object with the following structure:
        {
            "sample_size": integer,
            "tables": [
                {
                    "description": "Brief description of the table",
                    "percentages": [list of percentages],
                    "means": [list of means],
                    "standard_deviations": [list of standard deviations],
                    "min_values": [list of minimum values],
                    "max_values": [list of maximum values]
                }
            ]
        }
        
        Only include fields that are present in the paper. If a field is not present, omit it from the JSON.
        Ensure all numbers are extracted as floats or integers, not strings."""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def analyze_paper(self, pdf_path: str) -> Dict:
        """Analyze a paper using the LLM to extract statistical data."""
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Split text into chunks if it's too long (Gemini has token limits)
        max_chunk_length = 30000  # Gemini's approximate token limit
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        # Analyze each chunk and combine results
        all_results = []
        for chunk in chunks:
            try:
                # Create the prompt
                prompt = f"{self.system_prompt}\n\nAnalyze this text and extract the statistical data:\n\n{chunk}"
                
                # Generate response
                response = self.model.generate_content(prompt)
                
                # Parse the response
                try:
                    # Clean the response text to ensure it's valid JSON
                    response_text = response.text.strip()
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    
                    result = json.loads(response_text)
                    all_results.append(result)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON response: {str(e)}")
                    print(f"Raw response: {response_text}")
                    continue
                    
            except Exception as e:
                print(f"Warning: Error processing chunk: {str(e)}")
                continue
        
        # Combine results from all chunks
        combined_result = self._combine_results(all_results)
        return combined_result

    def _combine_results(self, results: List[Dict]) -> Dict:
        """Combine results from multiple chunks into a single coherent result."""
        if not results:
            return {"sample_size": None, "tables": []}
        
        # Use the first non-null sample size
        sample_size = next((r["sample_size"] for r in results if r.get("sample_size")), None)
        
        # Combine all tables
        all_tables = []
        for result in results:
            if "tables" in result:
                all_tables.extend(result["tables"])
        
        return {
            "sample_size": sample_size,
            "tables": all_tables
        }

    def process_papers_directory(self, directory: str) -> List[Dict]:
        """Process all PDF papers in a directory."""
        results = []
        for filename in os.listdir(directory):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(directory, filename)
                try:
                    result = self.analyze_paper(pdf_path)
                    result["filename"] = filename
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        return results 