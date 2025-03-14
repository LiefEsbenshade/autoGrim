from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class ImpossibleNumberResult:
    is_impossible: bool
    explanation: str
    reported_value: float
    sample_size: int
    type: str  # 'percentage', 'mean', or 'sd'

class ImpossibleNumberDetector:
    @staticmethod
    def check_percentage(percentage: float, sample_size: int) -> ImpossibleNumberResult:
        """
        Check if a reported percentage is possible given the sample size.
        A percentage is impossible if it can't be achieved with the given sample size.
        For example, 43% with 20 participants is impossible because 20 * 0.43 = 8.6,
        which can't be a whole number.
        """
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

    @staticmethod
    def check_mean_and_sd(mean: float, sd: float, min_val: float, max_val: float, sample_size: int) -> ImpossibleNumberResult:
        """
        Check if reported mean and standard deviation are possible given the data range.
        Uses Chebyshev's inequality and the fact that all values must be within the reported range.
        """
        # Check if mean is within the range
        if not (min_val <= mean <= max_val):
            return ImpossibleNumberResult(
                is_impossible=True,
                explanation=f"Mean {mean} is outside reported range [{min_val}, {max_val}]",
                reported_value=mean,
                sample_size=sample_size,
                type='mean'
            )
        
        # Check if standard deviation is possible
        # Using Chebyshev's inequality: at least 1-1/k² of values are within k standard deviations
        # For a normal distribution, about 99.7% of values are within 3 standard deviations
        max_possible_sd = (max_val - min_val) / 6  # Conservative estimate
        
        if sd > max_possible_sd:
            return ImpossibleNumberResult(
                is_impossible=True,
                explanation=f"Standard deviation {sd} is too large for the reported range [{min_val}, {max_val}]",
                reported_value=sd,
                sample_size=sample_size,
                type='sd'
            )
        
        return ImpossibleNumberResult(
            is_impossible=False,
            explanation=f"Mean {mean} and SD {sd} are possible within range [{min_val}, {max_val}]",
            reported_value=sd,
            sample_size=sample_size,
            type='sd'
        )

    @staticmethod
    def analyze_table(table_data: Dict[str, List[float]], sample_size: int) -> List[ImpossibleNumberResult]:
        """
        Analyze a table of data for impossible numbers.
        table_data should be a dictionary with keys like 'percentage', 'mean', 'sd', 'min', 'max'
        """
        results = []
        
        # Check percentages if present
        if 'percentage' in table_data:
            for percentage in table_data['percentage']:
                results.append(ImpossibleNumberDetector.check_percentage(percentage, sample_size))
        
        # Check means and standard deviations if present
        if all(key in table_data for key in ['mean', 'sd', 'min', 'max']):
            for mean, sd, min_val, max_val in zip(
                table_data['mean'],
                table_data['sd'],
                table_data['min'],
                table_data['max']
            ):
                results.append(ImpossibleNumberDetector.check_mean_and_sd(mean, sd, min_val, max_val, sample_size))
        
        return results 