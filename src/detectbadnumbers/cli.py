"""Command line interface for the detectbadnumbers package."""

import os
import sys
import logging
from typing import List, Dict
from dotenv import load_dotenv
from .llm_extractor import PaperAnalyzer, configure_llm
from .impossible_numbers import ImpossibleNumberDetector
import click

# Configure logging
logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging based on environment variables."""
    log_level = os.getenv("LOG_LEVEL", "WARNING")
    
    # Set root logger to WARNING
    logging.getLogger().setLevel(logging.WARNING)
    
    # Set all our module loggers to WARNING
    logging.getLogger("detectbadnumbers").setLevel(logging.WARNING)
    
    # Configure the basic format
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_papers_directory():
    """Create the papers directory if it doesn't exist."""
    if not os.path.exists('papers'):
        os.makedirs('papers')
        logging.info("Created 'papers' directory. Please add your PDF papers to this directory.")

@click.command()
@click.argument('papers_dir', type=click.Path(exists=True))
@click.option('--no-cache', is_flag=True, help='Disable caching of LLM results')
def main(papers_dir: str, no_cache: bool):
    """Analyze papers for impossible numbers in statistical data."""
    try:
        # Set up logging first
        setup_logging()
        
        # Load environment variables with override=True
        env_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(env_path, override=True)
        
        # Initialize Gemini model
        model = configure_llm()
        
        # Initialize analyzer with model
        analyzer = PaperAnalyzer(model=model, use_cache=not no_cache)
        
        # Process all PDFs in the directory
        results = analyzer.process_papers_directory(papers_dir)
        
        # Print results
        for result in results:
            print(f"\nAnalyzing {result['filename']}")
            print(f"Sample size: {result.get('sample_size', 'Unknown')}")
            print("=" * 80)
            
            # Analyze each statistical test
            for stat in result.get('text_statistics', []):
                print(f"\nStatistical Analysis: {stat.get('test_type', 'Unknown')}")
                print(f"Context: {stat.get('context', 'No context provided')}")
                print("-" * 80)
                
                # Check for impossible numbers
                detector = ImpossibleNumberDetector()
                issues = detector.analyze_text_statistics(stat, result.get('sample_size'))
                
                if issues:
                    print("\nImpossible numbers found:")
                    for issue in issues:
                        print(f"- {issue}")
                else:
                    print("\nNo impossible numbers found in these statistics.")
                
                print("-" * 80)
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 