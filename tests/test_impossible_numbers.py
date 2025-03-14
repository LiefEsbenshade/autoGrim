"""Tests for the impossible numbers detector module."""

import pytest
from detectbadnumbers.impossible_numbers import ImpossibleNumberDetector

def test_analyze_text_statistics_valid_t_test():
    """Test analysis of valid t-test statistics."""
    detector = ImpossibleNumberDetector()
    stat = {
        "test_type": "t-test",
        "sample_size": 30,
        "reported_statistics": {
            "test_statistic": "t(28) = 2.14",
            "p_value": "p < .05"
        }
    }
    
    issues = detector.analyze_text_statistics(stat, stat["sample_size"])
    assert not issues, "No issues should be found for valid statistics"

def test_analyze_text_statistics_impossible_df():
    """Test detection of impossible degrees of freedom."""
    detector = ImpossibleNumberDetector()
    stat = {
        "test_type": "t-test",
        "sample_size": 10,
        "reported_statistics": {
            "test_statistic": "t(28) = 2.14",  # Impossible df with n=10
            "p_value": "p < .05"
        }
    }
    
    issues = detector.analyze_text_statistics(stat, stat["sample_size"])
    assert issues, "Should detect impossible degrees of freedom"
    assert any("degrees of freedom" in str(issue).lower() for issue in issues)

def test_analyze_text_statistics_missing_sample_size():
    """Test handling of missing sample size."""
    detector = ImpossibleNumberDetector()
    stat = {
        "test_type": "t-test",
        "reported_statistics": {
            "test_statistic": "t(28) = 2.14",
            "p_value": "p < .05"
        }
    }
    
    issues = detector.analyze_text_statistics(stat, None)
    assert not issues, "Should handle missing sample size gracefully"

def test_analyze_text_statistics_f_test():
    """Test analysis of F-test statistics."""
    detector = ImpossibleNumberDetector()
    stat = {
        "test_type": "F-test",
        "sample_size": 60,
        "reported_statistics": {
            "test_statistic": "F(2, 57) = 3.16",
            "p_value": "p < .05"
        }
    }
    
    issues = detector.analyze_text_statistics(stat, stat["sample_size"])
    assert not issues, "No issues should be found for valid F-test"

def test_analyze_text_statistics_impossible_f_test():
    """Test detection of impossible F-test degrees of freedom."""
    detector = ImpossibleNumberDetector()
    stat = {
        "test_type": "F-test",
        "sample_size": 20,
        "reported_statistics": {
            "test_statistic": "F(2, 57) = 3.16",  # Impossible with n=20
            "p_value": "p < .05"
        }
    }
    
    issues = detector.analyze_text_statistics(stat, stat["sample_size"])
    assert issues, "Should detect impossible F-test degrees of freedom"
    assert any("degrees of freedom" in str(issue).lower() for issue in issues)

def test_analyze_text_statistics_effect_size():
    """Test analysis of effect size values."""
    detector = ImpossibleNumberDetector()
    stat = {
        "test_type": "t-test",
        "sample_size": 30,
        "reported_statistics": {
            "test_statistic": "t(28) = 2.14",
            "p_value": "p < .05",
            "effect_size": "Cohen's d = 5.2"  # Unusually large effect size
        }
    }
    
    issues = detector.analyze_text_statistics(stat, stat["sample_size"])
    assert issues, "Should detect suspicious effect size"
    assert any("effect size" in str(issue).lower() for issue in issues) 