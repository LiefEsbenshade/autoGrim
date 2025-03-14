# This is experimental vibe coding - everything, including the readme has been vibed

# autoGrim

A tool for automatically detecting impossible statistical values in scientific papers using the GRIM (Granularity-Related Inconsistency of Means) test and other statistical validity checks.

## Overview

This tool analyzes scientific papers to detect mathematically impossible statistical values, such as:
- Means that violate the GRIM test
- Impossible degrees of freedom given sample sizes
- Statistically impossible effect sizes
- Inconsistent sample sizes across analyses

The analysis is powered by Google's Gemini model for text extraction and a suite of statistical validity checks.

## Background

The GRIM test checks if reported means are mathematically possible given integer data. For example, with 20 participants rating on a 1-5 scale, a reported mean of 3.48 would be impossible since dividing any whole number by 20 must result in a number ending in .00, .05, .10, etc.

This project was initially developed to detect the GRIM violations found in [Experiment 4 of "Keep Your Fingers Crossed!: How Superstition Improves Performance"](https://journals.sagepub.com/doi/epub/10.1177/0956797610372631), a case discussed in ["How Junk Science Persists in Academia"](https://statmodeling.stat.columbia.edu/2022/04/22/how-junk-science-persists-in-academia-news-media-and-social-media-resistance-to-the-resistance/).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LiefEsbenshade/autoGrim.git
cd autoGrim
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```
Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Usage

1. Place your PDF papers in the `papers` directory

2. Run the analysis:
```bash
detectbadnumbers papers
```

Options:
- `--no-cache`: Force fresh analysis without using cached results
- `--output-dir PATH`: Specify output directory (default: 'output')

## Output

For each paper analyzed, the tool generates:
- A text report with all statistical tests and any detected issues
- A JSON file containing the raw extracted data
- Console output summarizing the findings

## Development

Run tests:
```bash
pip install -r tests/requirements-test.txt
pytest tests/ -v
```

For coverage report:
```bash
pytest tests/ --cov=detectbadnumbers -v
``` 