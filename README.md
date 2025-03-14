# Impossible Numbers Detector

This project analyzes scientific papers to detect impossible numbers in reported statistics. It checks for cases where reported percentages, means, and standard deviations are mathematically impossible given the reported sample size and data ranges.

The project implements the GRIM (Granularity-Related Inconsistency of Means) test, which checks if reported means are mathematically possible given a dataset of integer values. For example, with 20 participants, a reported mean of 3.48 would be impossible since dividing any whole number by 20 must result in a number ending in .00, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, or .95. [Learn more about the GRIM test](https://en.wikipedia.org/wiki/GRIM_test).

Initial testing of this tool is designed to detect the GRIM violations found in [Table 4 of "Keep Your Fingers Crossed!: How Superstition Improves Performance"](https://journals.sagepub.com/doi/epub/10.1177/0956797610372631), a case discussed in the post script of ["How Junk Science Persists in Academia, News Media, and Social Media: Resistance to the Resistance"](https://statmodeling.stat.columbia.edu/2022/04/22/how-junk-science-persists-in-academia-news-media-and-social-media-resistance-to-the-resistance/).

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