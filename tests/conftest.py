"""Test configuration and shared fixtures."""

import os
import pytest
import tempfile
import json
from typing import Dict

@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a temporary PDF file for testing."""
    pdf_path = tmp_path / "test.pdf"
    # Create an empty PDF file
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")
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
def output_dir(tmp_path):
    """Create a temporary output directory."""
    output_path = tmp_path / "output"
    output_path.mkdir()
    return str(output_path)

@pytest.fixture
def mock_env_file(tmp_path):
    """Create a mock .env file."""
    env_path = tmp_path / ".env"
    env_path.write_text("GOOGLE_API_KEY=test_api_key\nLOG_LEVEL=WARNING\n")
    return str(env_path)

@pytest.fixture
def cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    return str(cache_path) 