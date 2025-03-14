# Impossible Numbers Detector

This project analyzes scientific papers to detect impossible numbers in reported statistics. It checks for cases where reported percentages, means, and standard deviations are mathematically impossible given the reported sample size and data ranges.

## Features

- PDF paper analysis using Google's Gemini model
- Detection of impossible percentages based on sample size
- Analysis of means and standard deviations against reported ranges
- Automated data extraction from research papers

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```
You can get a free API key from: https://makersuite.google.com/app/apikey

4. Place research papers in the `papers` directory

## Usage

Run the main script:
```bash
python main.py
```

## Project Structure

- `papers/`: Directory for research papers
- `main.py`: Main script for running the analysis
- `impossible_numbers.py`: Core logic for detecting impossible numbers
- `llm_extractor.py`: LLM integration for extracting data from papers 