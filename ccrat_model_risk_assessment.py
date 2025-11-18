"""
Individual risk assessment submodel implementing NCI CCRAT methodology.

This module calculates personalized 5-year colorectal cancer risk based on
demographic characteristics using validated CCRAT algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class IndividualRiskAssessment:
    """
    Individual risk assessment component using CCRAT methodology.

    Implements NCI Colorectal Cancer Risk Assessment Tool algorithms for
    calculating personalized 5-year colorectal cancer risk based on age
    and demographic characteristics.
    """

    def __init__(self, config):
        """
        Initialize risk assessment component.

        Args:
            config: Model configuration object
        """
        self.config = config

        # CCRAT risk parameters
        self.ccrat_parameters = {
            'age_baseline_risk': {
                '45_49': 0.18, '50_54': 0.28, '55_59': 0.45,
                '60_64': 0.72, '65_69': 1.15, '70_75': 1.65
            },
            'gender_multipliers': {
                'male': 1.18, 'female': 0.85
            },
            'race_multipliers': {
                'white': 1.00, 'black': 1.28, 'hispanic': 0.82,
                'asian': 1.15, 'other': 0.92
            },
            'income_multipliers': {
                'under_20k': 1.25, '20k_35k': 1.12, '35k_50k': 1.03,
                '50k_75k': 0.96, 'over_75k': 0.88
            },
            'education_multipliers': {
                'less_than_hs': 1.20, 'hs_graduate': 1.08,
                'some_college': 0.95, 'bachelor_plus': 0.85
            }
        }

        # Screening effectiveness
        self.screening_effectiveness = 0.68  # 68% mortality reduction

        logger.info("Initialized Individual Risk Assessment component")

    def calculate_risk(
        self,
        age: str,
        gender: str,
        race: str,
        income: str,
        education: str
    ) -> Dict[str, float]:
        """
        Calculate CCRAT-style individual colorectal cancer risk.

        Args:
            age: Age group ('45_49', '50_54', etc.)
            gender: 'male' or 'female'
            race: 'white', 'black', 'hispanic', 'asian', 'other'
            income: Income level ('under_20k' to 'over_75k')
            education: Education level ('less_than_hs' to 'bachelor_plus')

        Returns:
            Dictionary with risk estimates:
                - unscreened_risk: 5-year risk without screening (%)
                - screened_risk: 5-year risk with screening (%)
                - screening_benefit: Risk reduction from screening (%)
                - risk_components: Individual multiplier values
        """
        # Get baseline risk for age group
        baseline_risk = self.ccrat_parameters['age_baseline_risk'].get(age, 0.5)

        # Apply demographic multipliers
        gender_mult = self.ccrat_parameters['gender_multipliers'].get(gender, 1.0)
        race_mult = self.ccrat_parameters['race_multipliers'].get(race, 1.0)
        income_mult = self.ccrat_parameters['income_multipliers'].get(income, 1.0)
        education_mult = self.ccrat_parameters['education_multipliers'].get(education, 1.0)

        # Calculate individual risk
        unscreened_risk = baseline_risk * gender_mult * race_mult * income_mult * education_mult
        screened_risk = unscreened_risk * (1 - self.screening_effectiveness)

        return {
            'unscreened_risk': unscreened_risk,
            'screened_risk': screened_risk,
            'screening_benefit': unscreened_risk - screened_risk,
            'risk_components': {
                'baseline_risk': baseline_risk,
                'gender_multiplier': gender_mult,
                'race_multiplier': race_mult,
                'income_multiplier': income_mult,
                'education_multiplier': education_mult
            }
        }

    def calculate_population_risks(
        self,
        population: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate risks for entire population DataFrame.

        Args:
            population: DataFrame with demographic columns

        Returns:
            DataFrame with added risk columns
        """
        risks = []
        for _, individual in population.iterrows():
            risk = self.calculate_risk(
                individual['age'],
                individual['gender'],
                individual['race'],
                individual['income'],
                individual['education']
            )
            risks.append(risk)

        risk_df = pd.DataFrame(risks)
        return pd.concat([population.reset_index(drop=True), risk_df], axis=1)

    def get_risk_distribution(
        self,
        population_risks: pd.DataFrame
    ) -> Dict:
        """
        Calculate summary statistics for population risk distribution.

        Args:
            population_risks: DataFrame with calculated risks

        Returns:
            Dictionary with distribution statistics
        """
        return {
            'mean_unscreened_risk': population_risks['unscreened_risk'].mean(),
            'median_unscreened_risk': population_risks['unscreened_risk'].median(),
            'std_unscreened_risk': population_risks['unscreened_risk'].std(),
            'min_risk': population_risks['unscreened_risk'].min(),
            'max_risk': population_risks['unscreened_risk'].max(),
            'percentile_25': population_risks['unscreened_risk'].quantile(0.25),
            'percentile_75': population_risks['unscreened_risk'].quantile(0.75),
            'screening_benefit_mean': population_risks['screening_benefit'].mean()
        }
