"""Tests for the LLM extractor module."""

import os
import pytest
import json
from unittest.mock import Mock, patch
from detectbadnumbers.llm_extractor import PaperAnalyzer, configure_llm
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

@pytest.fixture
def mock_model():
    """Create a mock LLM model."""
    model = Mock()
    model.generate_content.return_value = Mock(
        text=json.dumps({
            "text_statistics": [
                {
                    "test_type": "t-test",
                    "context": "Test comparison",
                    "sample_size": 30,
                    "reported_statistics": {
                        "test_statistic": "t(28) = 2.14",
                        "p_value": "p < .05"
                    }
                }
            ]
        })
    )
    return model

def test_configure_llm_missing_api_key():
    """Test LLM configuration with missing API key."""
    with pytest.raises(ValueError, match="GOOGLE_API_KEY.*not set"):
        with patch.dict(os.environ, clear=True):
            configure_llm()

def test_configure_llm_with_api_key():
    """Test LLM configuration with valid API key."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
        with patch("google.generativeai.configure") as mock_configure:
            with patch("google.generativeai.GenerativeModel") as mock_model:
                model = configure_llm()
                mock_configure.assert_called_once_with(api_key="test_key")
                mock_model.assert_called_once_with('gemini-1.5-pro-001')

def test_paper_analyzer_init(mock_model, test_cache_dir):
    """Test PaperAnalyzer initialization."""
    analyzer = PaperAnalyzer(mock_model, cache_dir=test_cache_dir)
    assert analyzer.model == mock_model
    assert analyzer.cache_dir == test_cache_dir
    assert analyzer.use_cache is True

def test_paper_analyzer_cache_handling(mock_model, test_cache_dir, sample_pdf_path):
    """Test cache handling in PaperAnalyzer."""
    analyzer = PaperAnalyzer(mock_model, cache_dir=test_cache_dir)
    
    # First run - should use model
    result1 = analyzer.analyze_paper(sample_pdf_path)
    assert mock_model.generate_content.called
    mock_model.generate_content.reset_mock()
    
    # Second run - should use cache
    result2 = analyzer.analyze_paper(sample_pdf_path)
    assert not mock_model.generate_content.called
    assert result1 == result2

def test_paper_analyzer_no_cache(mock_model, test_cache_dir, sample_pdf_path):
    """Test PaperAnalyzer with cache disabled."""
    analyzer = PaperAnalyzer(mock_model, cache_dir=test_cache_dir, use_cache=False)
    
    # First run
    analyzer.analyze_paper(sample_pdf_path)
    assert mock_model.generate_content.called
    mock_model.generate_content.reset_mock()
    
    # Second run - should still use model
    analyzer.analyze_paper(sample_pdf_path)
    assert mock_model.generate_content.called

def test_paper_analyzer_process_directory(mock_model, tmp_path):
    """Test processing of multiple papers in a directory."""
    # Create test PDFs
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    
    # Create proper PDFs with reportlab
    for i in range(2):
        pdf_path = papers_dir / f"test{i+1}.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawString(100, 750, f"Sample statistical text for testing {i+1}.")
        c.drawString(100, 730, "The t-test revealed significant differences (t(28) = 2.14, p < .05).")
        c.drawString(100, 710, "Effect size was moderate (Cohen's d = 0.78).")
        c.save()

    analyzer = PaperAnalyzer(mock_model, cache_dir=str(tmp_path / "test_cache"))
    results = analyzer.process_papers_directory(str(papers_dir))
    
    assert len(results) == 2
    assert all(isinstance(r, dict) for r in results)
    assert mock_model.generate_content.call_count == 2

def test_paper_analyzer_text_extraction(mock_model, sample_pdf_path):
    """Test PDF text extraction."""
    analyzer = PaperAnalyzer(mock_model)
    text = analyzer.extract_text_from_pdf(sample_pdf_path)
    assert isinstance(text, str)

def test_paper_analyzer_invalid_pdf(mock_model, tmp_path):
    """Test handling of invalid PDF files."""
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_text("Not a PDF file")
    
    analyzer = PaperAnalyzer(mock_model)
    with pytest.raises(Exception):
        analyzer.extract_text_from_pdf(str(invalid_pdf))

def test_paper_analyzer_combine_results(mock_model):
    """Test combining results from multiple chunks."""
    analyzer = PaperAnalyzer(mock_model)
    results = [
        {
            "text_statistics": [{"id": 1}],
            "tables": [{"id": "A"}]
        },
        {
            "text_statistics": [{"id": 2}],
            "tables": [{"id": "B"}]
        }
    ]
    
    combined = analyzer._combine_results(results)
    assert len(combined["text_statistics"]) == 2
    assert len(combined["tables"]) == 2 