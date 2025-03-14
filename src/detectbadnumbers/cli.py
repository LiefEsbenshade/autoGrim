"""Command line interface for the detectbadnumbers package."""

import os
import sys
import logging
from typing import List, Dict
from dotenv import load_dotenv
from .llm_extractor import PaperAnalyzer, configure_llm
from .impossible_numbers import ImpossibleNumberDetector
import click
import json
from datetime import datetime

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

def save_analysis_results(results: Dict, output_dir: str, filename: str):
    """Save analysis results to a file in the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(filename)[0]
    output_file = os.path.join(output_dir, f"{base_name}_analysis_{timestamp}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Analysis Results for {filename}\n")
        f.write("=" * 80 + "\n\n")
        
        for stat in results.get('text_statistics', []):
            f.write(f"Statistical Analysis: {stat.get('test_type', 'Unknown')}\n")
            f.write(f"Context: {stat.get('context', 'No context provided')}\n")
            
            if 'sample_size' in stat:
                f.write(f"Sample Size (N): {stat['sample_size']}\n")
            
            if 'reported_statistics' in stat:
                f.write("\nReported Statistics:\n")
                stats = stat['reported_statistics']
                if isinstance(stats, dict):
                    for key, value in stats.items():
                        f.write(f"- {key}: {value}\n")
                elif isinstance(stats, list):
                    for item in stats:
                        f.write(f"- {item}\n")
                else:
                    f.write(f"- {stats}\n")
            
            f.write("-" * 80 + "\n")
            
            # Write analysis results
            detector = ImpossibleNumberDetector()
            issues = detector.analyze_text_statistics(stat, stat.get('sample_size'))
            
            if issues:
                f.write("\nImpossible numbers found:\n")
                for issue in issues:
                    f.write(f"- {issue}\n")
            else:
                f.write("\nNo impossible numbers found in these statistics.\n")
            
            f.write("-" * 80 + "\n\n")
    
    # Also save the raw JSON data
    json_file = os.path.join(output_dir, f"{base_name}_analysis_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    return output_file, json_file

@click.command()
@click.argument('papers_dir', type=click.Path(exists=True))
@click.option('--no-cache', is_flag=True, help='Disable caching of LLM results')
@click.option('--output-dir', '-o', type=click.Path(), default='output', help='Directory to save analysis results')
def main(papers_dir: str, no_cache: bool, output_dir: str):
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
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Print and save results
        for result in results:
            print(f"\nAnalyzing {result['filename']}")
            print("=" * 80)
            
            # Save results to files
            txt_file, json_file = save_analysis_results(result, output_dir, result['filename'])
            print(f"\nResults saved to:")
            print(f"- Text report: {txt_file}")
            print(f"- JSON data: {json_file}")
            
            # Print results to console as before
            for stat in result.get('text_statistics', []):
                print(f"\nStatistical Analysis: {stat.get('test_type', 'Unknown')}")
                print(f"Context: {stat.get('context', 'No context provided')}")
                
                if 'sample_size' in stat:
                    print(f"Sample Size (N): {stat['sample_size']}")
                
                if 'reported_statistics' in stat:
                    print("\nReported Statistics:")
                    stats = stat['reported_statistics']
                    if isinstance(stats, dict):
                        for key, value in stats.items():
                            print(f"- {key}: {value}")
                    elif isinstance(stats, list):
                        for item in stats:
                            print(f"- {item}")
                    else:
                        print(f"- {stats}")
                
                print("-" * 80)
                
                # Check for impossible numbers
                detector = ImpossibleNumberDetector()
                issues = detector.analyze_text_statistics(stat, stat.get('sample_size'))
                
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