"""
Screening access submodel for predicting colorectal cancer screening utilization.

This module predicts screening utilization based on insurance status and
demographic characteristics, incorporating evidence-based screening rate differentials.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ScreeningAccessModel:
    """
    Screening access component.

    Predicts colorectal cancer screening utilization based on insurance status
    and demographic factors using evidence-based screening rate differentials.
    """

    def __init__(self, config):
        """
        Initialize screening access component.

        Args:
            config: Model configuration object
        """
        self.config = config

        # Base screening rates by insurance type
        self.screening_rates = {
            'insurance': {
                'medicaid': 0.466,
                'uninsured': 0.282,
                'private': 0.598
            },
            'age': {
                '45_49': 0.420, '50_54': 0.520, '55_59': 0.580,
                '60_64': 0.620, '65_69': 0.650, '70_75': 0.630
            },
            'race_ethnicity': {
                'white': 0.598, 'black': 0.602, 'hispanic': 0.465,
                'asian': 0.517, 'other': 0.557
            },
            'income': {
                'under_20k': 0.446, '20k_35k': 0.520, '35k_50k': 0.580,
                '50k_75k': 0.620, 'over_75k': 0.645
            },
            'education': {
                'less_than_hs': 0.468, 'hs_graduate': 0.550,
                'some_college': 0.580, 'bachelor_plus': 0.620
            },
            'gender': {
                'male': 0.510, 'female': 0.570
            }
        }

        logger.info("Initialized Screening Access Model component")

    def predict_utilization(
        self,
        population: pd.DataFrame,
        insurance_transitions: Dict
    ) -> Dict:
        """
        Predict screening utilization for population.

        Args:
            population: DataFrame with demographic data
            insurance_transitions: Dictionary with insurance transition results

        Returns:
            Dictionary with screening utilization predictions
        """
        # Merge insurance transition data
        pop_with_insurance = insurance_transitions['population_with_transitions']

        # Calculate screening rates before and after transition
        screening_before = []
        screening_after = []

        for _, individual in pop_with_insurance.iterrows():
            # Screening rate while on Medicaid
            rate_medicaid = self._calculate_demographic_screening_rate(
                individual, 'medicaid'
            )
            screening_before.append(rate_medicaid)

            # Screening rate if loses coverage
            if individual.get('loses_coverage', False):
                rate_after = self._calculate_demographic_screening_rate(
                    individual, 'uninsured'
                )
            else:
                rate_after = rate_medicaid

            screening_after.append(rate_after)

        pop_with_insurance = pop_with_insurance.copy()
        pop_with_insurance['screening_rate_before'] = screening_before
        pop_with_insurance['screening_rate_after'] = screening_after
        pop_with_insurance['screening_reduction'] = (
            pop_with_insurance['screening_rate_before'] - 
            pop_with_insurance['screening_rate_after']
        )

        # Calculate aggregate statistics
        total_reduction = pop_with_insurance['screening_reduction'].sum()
        avg_rate_before = pop_with_insurance['screening_rate_before'].mean()
        avg_rate_after = pop_with_insurance['screening_rate_after'].mean()

        return {
            'reduction': total_reduction,
            'avg_rate_before': avg_rate_before,
            'avg_rate_after': avg_rate_after,
            'population_with_screening': pop_with_insurance
        }

    def _calculate_demographic_screening_rate(
        self,
        individual: pd.Series,
        insurance_status: str
    ) -> float:
        """Calculate demographic-adjusted screening rate for individual."""
        base_rate = self.screening_rates['insurance'][insurance_status]

        # Apply demographic adjustments
        adjustments = []

        # Age adjustment
        age_rate = self.screening_rates['age'].get(individual['age'], base_rate)
        adjustments.append(age_rate / base_rate)

        # Race adjustment
        race_rate = self.screening_rates['race_ethnicity'].get(individual['race'], base_rate)
        adjustments.append(race_rate / base_rate)

        # Income adjustment
        income_rate = self.screening_rates['income'].get(individual['income'], base_rate)
        adjustments.append(income_rate / base_rate)

        # Apply geometric mean of adjustments
        if adjustments:
            adjustment_factor = np.prod(adjustments) ** (1/len(adjustments))
            adjusted_rate = base_rate * adjustment_factor
        else:
            adjusted_rate = base_rate

        return min(0.90, max(0.10, adjusted_rate))  # Bound between 10% and 90%
