"""Module for detecting impossible numbers in statistical data."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import math
import re

@dataclass
class ImpossibleNumberResult:
    is_impossible: bool
    explanation: str
    reported_value: float
    sample_size: int
    type: str  # 'percentage', 'mean', 'sd', 'df', or 'effect_size'

class ImpossibleNumberDetector:
    def __init__(self):
        """Initialize the impossible number detector."""
        self.effect_size_thresholds = {
            "Cohen's d": 2.0,  # Values above this are suspicious
            "eta squared": 0.5,  # Values above this are suspicious
            "r": 1.0,  # Values above this are impossible
        }

    @staticmethod
    def check_percentage(percentage: float, sample_size: int) -> ImpossibleNumberResult:
        """
        Check if a reported percentage is possible given the sample size.
        A percentage is impossible if it can't be achieved with the given sample size.
        For example, 43% with 20 participants is impossible because 20 * 0.43 = 8.6,
        which can't be a whole number.
        """
        # Check for None values
        if percentage is None or sample_size is None:
            return ImpossibleNumberResult(
                is_impossible=True,
                explanation=f"Invalid values: percentage={percentage}, sample_size={sample_size}",
                reported_value=0.0,
                sample_size=0,
                type='percentage'
            )
            
        # Convert to numeric types
        try:
            percentage = float(percentage)
            sample_size = int(sample_size)
        except (TypeError, ValueError):
            return ImpossibleNumberResult(
                is_impossible=True,
                explanation=f"Non-numeric values: percentage={percentage}, sample_size={sample_size}",
                reported_value=0.0,
                sample_size=0,
                type='percentage'
            )
        
        if not (0 <= percentage <= 100):
            return ImpossibleNumberResult(
                is_impossible=True,
                explanation=f"Percentage {percentage} is outside valid range [0, 100]",
                reported_value=percentage,
                sample_size=sample_size,
                type='percentage'
            )
        
        # Calculate how many participants this percentage would represent
        actual_count = sample_size * (percentage / 100)
        
        # Check if the count is a whole number
        if not np.isclose(actual_count, round(actual_count), rtol=1e-10):
            return ImpossibleNumberResult(
                is_impossible=True,
                explanation=f"Percentage {percentage}% with {sample_size} participants would require {actual_count} participants, which is not possible",
                reported_value=percentage,
                sample_size=sample_size,
                type='percentage'
            )
        
        return ImpossibleNumberResult(
            is_impossible=False,
            explanation=f"Percentage {percentage}% is possible with {sample_size} participants",
            reported_value=percentage,
            sample_size=sample_size,
            type='percentage'
        )

    def check_mean_and_sd(self, mean: float, sd: float) -> List[str]:
        """
        Check if a mean and standard deviation pair is impossible.
        Returns a list of strings describing any impossible numbers found.
        """
        errors = []
        
        # Check for None values
        if mean is None or sd is None:
            return [f"Invalid values: mean={mean}, sd={sd}"]
        
        # Convert to float to ensure numeric comparisons work
        try:
            mean = float(mean)
            sd = float(sd)
        except (TypeError, ValueError):
            return [f"Non-numeric values: mean={mean}, sd={sd}"]
        
        # Check for negative standard deviation
        if sd < 0:
            errors.append(f"Standard deviation {sd} is negative")
            
        # Check for zero standard deviation with non-zero mean
        if sd == 0 and mean != 0:
            errors.append(f"Standard deviation is 0 but mean is {mean}")
            
        # For Likert scales (1-7 or 1-9)
        if 1 <= mean <= 7:
            max_sd = math.sqrt((7-1)**2 / 4)  # Maximum possible SD for 1-7 scale
            if sd > max_sd:
                errors.append(f"Standard deviation {sd} is too large for a 1-7 scale (max possible: {max_sd:.2f})")
        elif 1 <= mean <= 9:
            max_sd = math.sqrt((9-1)**2 / 4)  # Maximum possible SD for 1-9 scale
            if sd > max_sd:
                errors.append(f"Standard deviation {sd} is too large for a 1-9 scale (max possible: {max_sd:.2f})")
                
        # For percentages (0-100)
        if 0 <= mean <= 100:
            max_sd = math.sqrt((100-0)**2 / 4)  # Maximum possible SD for percentage
            if sd > max_sd:
                errors.append(f"Standard deviation {sd} is too large for a percentage (max possible: {max_sd:.2f})")
                
        # For positive-only values
        if mean > 0:
            # Skip the check if it looks like standardized data (z-scores)
            is_likely_zscore = -3 <= mean <= 3 and 0.5 <= sd <= 2.0
            if not is_likely_zscore and sd > mean * 3:
                errors.append(f"Standard deviation {sd} is unusually large compared to mean {mean}")
                
        return errors

    @staticmethod
    def analyze_table(table_data: Dict[str, List[float]], sample_size: int) -> List[ImpossibleNumberResult]:
        """
        Analyze a table of data for impossible numbers.
        table_data should be a dictionary with keys like 'percentage', 'mean', 'sd', 'min', 'max'
        """
        results = []
        
        # Check percentages if present
        if 'percentages' in table_data:
            for percentage in table_data['percentages']:
                results.append(ImpossibleNumberDetector.check_percentage(percentage, sample_size))
        
        # Check means and standard deviations if present
        if all(key in table_data for key in ['means', 'standard_deviations', 'min_values', 'max_values']):
            for mean, sd, min_val, max_val in zip(
                table_data['means'],
                table_data['standard_deviations'],
                table_data['min_values'],
                table_data['max_values']
            ):
                results.append(ImpossibleNumberDetector.check_mean_and_sd(mean, sd))
        
        return results

    @staticmethod
    def analyze_text_statistics(text_stats: Dict, sample_size: int) -> List[ImpossibleNumberResult]:
        """
        Analyze statistical data from text descriptions for impossible numbers.
        text_stats should be a dictionary containing statistical information from text.
        """
        results = []
        
        # Check means and standard deviations for each condition
        if 'conditions' in text_stats:
            for condition in text_stats['conditions']:
                if 'mean' in condition and 'sd' in condition:
                    mean = condition['mean']
                    sd = condition['sd']
                    name = condition.get('name', 'Unnamed condition')
                    
                    # For text statistics, we don't have min/max values, so we'll use a conservative range
                    # based on the mean and standard deviation
                    min_val = mean - (3 * sd)  # 3 standard deviations below mean
                    max_val = mean + (3 * sd)  # 3 standard deviations above mean
                    
                    result = ImpossibleNumberDetector.check_mean_and_sd(mean, sd)
                    
                    # Add condition context to the explanation
                    if result:
                        result_str = ", ".join(result)
                        result_str = f"In {name}: {result_str}"
                        results.append(ImpossibleNumberResult(
                            is_impossible=True,
                            explanation=result_str,
                            reported_value=mean,
                            sample_size=sample_size,
                            type='mean'
                        ))
        
        return results

    def _is_possible_grim_mean(self, mean: float, n: int) -> bool:
        """
        Check if a mean could possibly be produced by n participants when the measure is a count.
        Returns True if the mean is possible, False otherwise.
        """
        # Calculate the sum of scores (should be an integer since it's a count)
        total = mean * n
        
        # Try rounding up and down since we don't know which way the original sum was rounded
        total_up = math.ceil(total)
        total_down = math.floor(total)
        
        # Calculate what means these totals would produce
        mean_up = total_up / n
        mean_down = total_down / n
        
        # Check if either mean matches the reported mean (within floating point precision)
        return np.isclose(mean, mean_up, rtol=1e-10) or np.isclose(mean, mean_down, rtol=1e-10)

    def check_grim_test(self, conditions: List[Dict], total_n: int, is_count_data: bool = False) -> List[str]:
        """
        Perform a GRIM test on a set of conditions with a known total sample size.
        This checks if the reported means could possibly be produced by any combination
        of sample sizes that sum to total_n.
        
        Args:
            conditions: List of conditions, each with 'name' and 'mean'
            total_n: Total sample size across all conditions
            is_count_data: Whether the measure is count data (must be integers)
        
        Returns:
            List of error messages if GRIM test fails
        """
        if not is_count_data or len(conditions) != 2:
            return []  # Only handle count data with exactly 2 conditions for now
            
        errors = []
        cond1, cond2 = conditions
        mean1, mean2 = cond1['mean'], cond2['mean']
        name1, name2 = cond1.get('name', 'Condition 1'), cond2.get('name', 'Condition 2')
        
        # Try all possible sample size combinations
        possible_combination_found = False
        for n1 in range(1, total_n):
            n2 = total_n - n1
            if self._is_possible_grim_mean(mean1, n1) and self._is_possible_grim_mean(mean2, n2):
                possible_combination_found = True
                break
        
        if not possible_combination_found:
            errors.append(
                f"GRIM test failure: The means {mean1} ({name1}) and {mean2} ({name2}) "
                f"cannot both be produced by any combination of sample sizes that sum to {total_n}. "
                "Since this is count data, the sum of scores in each condition must be an integer."
            )
        
        return errors

    def check_degrees_of_freedom(self, test_type: str, df_str: str, sample_size: int) -> Optional[str]:
        """
        Check if the reported degrees of freedom are possible given the sample size.
        Returns an error message if impossible, None otherwise.
        """
        if test_type == "t-test":
            # Extract df from format like "t(28) = 2.14"
            match = re.search(r't\((\d+)\)', df_str)
            if match:
                df = int(match.group(1))
                # For independent t-test, df = n1 + n2 - 2
                # For paired t-test, df = n - 1
                # So df should always be less than total n
                if df >= sample_size:
                    return f"Degrees of freedom ({df}) cannot be larger than or equal to sample size ({sample_size})"
        elif test_type == "F-test":
            # Extract df from format like "F(2, 57) = 3.16"
            match = re.search(r'F\((\d+),\s*(\d+)\)', df_str)
            if match:
                df1, df2 = int(match.group(1)), int(match.group(2))
                # df2 is the error df, which should be less than sample size
                if df2 >= sample_size:
                    return f"Error degrees of freedom ({df2}) cannot be larger than or equal to sample size ({sample_size})"
                # df1 is the factor df, which should be less than sample size
                if df1 >= sample_size:
                    return f"Factor degrees of freedom ({df1}) cannot be larger than or equal to sample size ({sample_size})"
        return None

    def check_effect_size(self, effect_size_str: str) -> Optional[str]:
        """
        Check if the reported effect size is suspiciously large or impossible.
        Returns an error message if suspicious/impossible, None otherwise.
        """
        if not effect_size_str:
            return None
            
        # Extract effect size type and value
        for effect_type, threshold in self.effect_size_thresholds.items():
            if effect_type.lower() in effect_size_str.lower():
                try:
                    value = float(re.search(r'[-+]?\d*\.?\d+', effect_size_str).group())
                    if abs(value) > threshold:
                        return f"Suspicious effect size: {effect_type} of {value} is {'impossible' if threshold == 1.0 else 'unusually large'} (threshold: {threshold})"
                except (AttributeError, ValueError):
                    pass
        return None

    def analyze_text_statistics(self, text_stat: Dict, sample_size: Optional[int] = None) -> List[str]:
        """
        Analyze text statistics for impossible numbers.
        Returns a list of strings describing any impossible numbers found.
        """
        results = []
        
        # Check test statistics and degrees of freedom
        if 'reported_statistics' in text_stat:
            stats = text_stat['reported_statistics']
            test_type = text_stat.get('test_type')
            
            # Check degrees of freedom
            if sample_size and 'test_statistic' in stats:
                df_error = self.check_degrees_of_freedom(test_type, stats['test_statistic'], sample_size)
                if df_error:
                    results.append(df_error)
            
            # Check effect size
            if 'effect_size' in stats:
                effect_error = self.check_effect_size(stats['effect_size'])
                if effect_error:
                    results.append(effect_error)
        
        # Check means and standard deviations
        conditions = text_stat.get('conditions', [])
        for condition in conditions:
            if 'mean' in condition and 'sd' in condition:
                mean = condition['mean']
                sd = condition['sd']
                name = condition.get('name', 'Unknown condition')
                
                errors = self.check_mean_and_sd(mean, sd)
                if errors:
                    for error in errors:
                        results.append(f"{name}: {error}")
        
        # Check for GRIM test failures
        context = text_stat.get('context')
        if context:  # Only check if context exists and is not None
            context_lower = context.lower()
            is_count_data = any(word in context_lower for word in ['count', 'number of', 'correctly identified', 'solved'])
            if is_count_data and len(conditions) == 2 and sample_size:
                grim_errors = self.check_grim_test(conditions, sample_size, is_count_data=True)
                results.extend(grim_errors)
        
        return results

    def analyze_table(self, table_data: Dict) -> List[str]:
        """
        Analyze a table of data for impossible numbers.
        Returns a list of strings describing any impossible numbers found.
        """
        results = []
        
        # Add table analysis logic here if needed
        
        return results 