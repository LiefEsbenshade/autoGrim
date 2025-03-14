"""Test configuration and shared fixtures."""

import os
import pytest
import tempfile
import json
from typing import Dict
import shutil
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a temporary PDF file for testing."""
    pdf_path = tmp_path / "test.pdf"
    
    # Create a simple PDF with some text
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 750, "Sample statistical text for testing.")
    c.drawString(100, 730, "The t-test revealed significant differences (t(28) = 2.14, p < .05).")
    c.drawString(100, 710, "Effect size was moderate (Cohen's d = 0.78).")
    c.save()
    
    return str(pdf_path)

@pytest.fixture
def sample_statistics() -> Dict:
    """Return sample statistical data for testing."""
    return {
        "text_statistics": [
            {
                "test_type": "t-test",
                "context": "Comparison of group A and B",
                "sample_size": 30,
                "reported_statistics": {
                    "test_statistic": "t(28) = 2.14",
                    "p_value": "p < .05",
                    "effect_size": "Cohen's d = 0.78",
                    "means": {
                        "group1": "M = 3.45, SD = 0.67",
                        "group2": "M = 2.98, SD = 0.71"
                    }
                }
            }
        ]
    }

@pytest.fixture
def test_output_dir(tmp_path):
    """Create a temporary output directory that gets cleaned up after the test."""
    output_path = tmp_path / "test_output"
    output_path.mkdir()
    yield str(output_path)
    # Cleanup happens automatically as tmp_path is cleaned up by pytest

@pytest.fixture
def test_env_file(tmp_path):
    """Create a temporary .env file that gets cleaned up after the test."""
    env_path = tmp_path / ".env"
    env_path.write_text("GOOGLE_API_KEY=test_api_key\nLOG_LEVEL=WARNING\n")
    yield str(env_path)
    # Cleanup happens automatically as tmp_path is cleaned up by pytest

@pytest.fixture
def test_cache_dir(tmp_path):
    """Create a temporary cache directory that gets cleaned up after the test."""
    cache_path = tmp_path / "test_cache"
    cache_path.mkdir()
    yield str(cache_path)
    # Cleanup happens automatically as tmp_path is cleaned up by pytest

@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically clean up any test artifacts after each test."""
    yield  # Run the test
    # Clean up any stray directories that might have been created
    paths_to_cleanup = ['output', '.pytest_cache']
    for path in paths_to_cleanup:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path) 