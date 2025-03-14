"""Tests for the CLI module."""

import os
import pytest
import json
from click.testing import CliRunner
from unittest.mock import Mock, patch
from detectbadnumbers.cli import main, save_analysis_results

@pytest.fixture
def cli_runner():
    """Create a Click CLI runner."""
    return CliRunner()

@pytest.fixture
def mock_analyzer():
    """Create a mock PaperAnalyzer."""
    analyzer = Mock()
    analyzer.process_papers_directory.return_value = [{
        "filename": "test.pdf",
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
    }]
    return analyzer

def test_main_success(cli_runner, tmp_path, mock_analyzer):
    """Test successful execution of the main CLI command."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    output_dir = tmp_path / "test_output"
    
    with patch("detectbadnumbers.cli.PaperAnalyzer", return_value=mock_analyzer):
        with patch("detectbadnumbers.cli.configure_llm"):
            result = cli_runner.invoke(main, [
                str(papers_dir),
                "--output-dir", str(output_dir)
            ])
    
    assert result.exit_code == 0
    assert "Analyzing test.pdf" in result.output
    assert mock_analyzer.process_papers_directory.called

def test_main_missing_papers_dir(cli_runner):
    """Test CLI with non-existent papers directory."""
    result = cli_runner.invoke(main, ["nonexistent_dir"])
    assert result.exit_code != 0
    assert "Error" in result.output

def test_main_no_cache_option(cli_runner, tmp_path, mock_analyzer):
    """Test CLI with --no-cache option."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    
    with patch("detectbadnumbers.cli.PaperAnalyzer", return_value=mock_analyzer):
        with patch("detectbadnumbers.cli.configure_llm"):
            result = cli_runner.invoke(main, [
                str(papers_dir),
                "--no-cache"
            ])
    
    assert result.exit_code == 0
    assert mock_analyzer.process_papers_directory.called

def test_save_analysis_results(tmp_path, sample_statistics):
    """Test saving analysis results to files."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    
    txt_file, json_file = save_analysis_results(
        sample_statistics,
        str(output_dir),
        "test.pdf"
    )
    
    # Check text file
    assert os.path.exists(txt_file)
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()
        assert "Analysis Results for test.pdf" in content
        assert "Statistical Analysis: t-test" in content
        assert "Sample Size (N): 30" in content
    
    # Check JSON file
    assert os.path.exists(json_file)
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        assert data == sample_statistics

def test_save_analysis_results_creates_dir(tmp_path, sample_statistics):
    """Test that save_analysis_results creates output directory if needed."""
    output_dir = tmp_path / "nonexistent"
    
    txt_file, json_file = save_analysis_results(
        sample_statistics,
        str(output_dir),
        "test.pdf"
    )
    
    assert os.path.exists(output_dir)
    assert os.path.exists(txt_file)
    assert os.path.exists(json_file)

def test_main_with_env_file(cli_runner, tmp_path, mock_analyzer, test_env_file):
    """Test CLI with environment file."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    
    with patch("detectbadnumbers.cli.PaperAnalyzer", return_value=mock_analyzer):
        with patch("detectbadnumbers.cli.configure_llm"):
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
                result = cli_runner.invoke(main, [str(papers_dir)])
    
    assert result.exit_code == 0
    assert mock_analyzer.process_papers_directory.called 