import os
from typing import List, Dict
from llm_extractor import PaperAnalyzer
from impossible_numbers import ImpossibleNumberDetector

def create_papers_directory():
    """Create the papers directory if it doesn't exist."""
    if not os.path.exists('papers'):
        os.makedirs('papers')
        print("Created 'papers' directory. Please add your PDF papers to this directory.")

def analyze_papers():
    """Main function to analyze papers and detect impossible numbers."""
    # Initialize components
    paper_analyzer = PaperAnalyzer()
    detector = ImpossibleNumberDetector()
    
    # Process all papers
    results = paper_analyzer.process_papers_directory('papers')
    
    # Analyze results
    for paper_result in results:
        filename = paper_result.get('filename', 'Unknown file')
        sample_size = paper_result.get('sample_size')
        
        if not sample_size:
            print(f"\nWarning: No sample size found in {filename}")
            continue
        
        print(f"\nAnalyzing {filename} (Sample size: {sample_size})")
        print("-" * 50)
        
        for table in paper_result.get('tables', []):
            print(f"\nTable: {table.get('description', 'Unnamed table')}")
            
            # Prepare data for analysis
            table_data = {
                'percentage': table.get('percentages', []),
                'mean': table.get('means', []),
                'sd': table.get('standard_deviations', []),
                'min': table.get('min_values', []),
                'max': table.get('max_values', [])
            }
            
            # Analyze the table
            analysis_results = detector.analyze_table(table_data, sample_size)
            
            # Report findings
            impossible_results = [r for r in analysis_results if r.is_impossible]
            if impossible_results:
                print("\nFound impossible numbers:")
                for result in impossible_results:
                    print(f"- {result.explanation}")
            else:
                print("\nNo impossible numbers found in this table.")

def main():
    """Main entry point."""
    print("Impossible Numbers Detector")
    print("=" * 50)
    
    # Check for Google API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables")
        print("Please create a .env file with your Google API key")
        print("You can get a free API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Create papers directory if needed
    create_papers_directory()
    
    # Check if papers directory is empty
    if not os.listdir('papers'):
        print("\nNo papers found in the 'papers' directory.")
        print("Please add your PDF papers to the 'papers' directory and run again.")
        return
    
    # Analyze papers
    analyze_papers()

if __name__ == "__main__":
    main() 