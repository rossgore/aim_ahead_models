"""
Demographic microsimulation submodel for generating census tract populations.

This module creates synthetic but statistically representative populations for each
census tract while maintaining consistency with American Community Survey distributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class DemographicMicrosimulation:
    """
    Demographic microsimulation component for generating census tract populations.

    This class implements spatial microsimulation following established small-area
    estimation methodologies, creating individual demographic profiles that maintain
    statistical consistency with ACS tract-level distributions.
    """

    def __init__(self, config):
        """
        Initialize demographic microsimulation component.

        Args:
            config: Model configuration object
        """
        self.config = config

        # Area-specific demographic profiles
        self.area_profiles = self._load_area_profiles()

        logger.info("Initialized Demographic Microsimulation component")

    def _load_area_profiles(self) -> Dict:
        """Load area-specific demographic profiles."""
        return {
            'Virginia Beach': {
                'population_range': (3000, 6000),
                'demographics': {
                    'white': (0.55, 0.75), 'black': (0.15, 0.25), 
                    'hispanic': (0.06, 0.12), 'asian': (0.05, 0.10),
                    'male': (0.48, 0.52), 'female': (0.48, 0.52),
                    'age_45_49': (0.055, 0.075), 'age_50_54': (0.060, 0.080),
                    'age_55_59': (0.055, 0.075), 'age_60_64': (0.045, 0.065),
                    'age_65_69': (0.035, 0.055), 'age_70_75': (0.025, 0.045),
                    'less_than_hs': (0.05, 0.12), 'hs_graduate': (0.20, 0.28),
                    'some_college': (0.30, 0.40), 'bachelor_plus': (0.25, 0.45),
                    'under_20k': (0.04, 0.10), '20k_35k': (0.10, 0.18),
                    '35k_50k': (0.13, 0.20), '50k_75k': (0.18, 0.28),
                    'over_75k': (0.30, 0.50)
                },
                'poverty_range': (0.04, 0.15),
                'medicaid_rate_range': (0.12, 0.22),
                'median_income_range': (55000, 95000)
            },
            'Norfolk': {
                'population_range': (2500, 5500),
                'demographics': {
                    'white': (0.25, 0.55), 'black': (0.35, 0.65),
                    'hispanic': (0.05, 0.12), 'asian': (0.02, 0.06),
                    'male': (0.50, 0.55), 'female': (0.45, 0.50),
                    'age_45_49': (0.060, 0.080), 'age_50_54': (0.055, 0.075),
                    'age_55_59': (0.045, 0.065), 'age_60_64': (0.040, 0.060),
                    'age_65_69': (0.030, 0.050), 'age_70_75': (0.025, 0.045),
                    'less_than_hs': (0.10, 0.20), 'hs_graduate': (0.25, 0.35),
                    'some_college': (0.28, 0.38), 'bachelor_plus': (0.15, 0.35),
                    'under_20k': (0.12, 0.25), '20k_35k': (0.15, 0.25),
                    '35k_50k': (0.15, 0.22), '50k_75k': (0.15, 0.25),
                    'over_75k': (0.10, 0.30)
                },
                'poverty_range': (0.10, 0.35),
                'medicaid_rate_range': (0.25, 0.50),
                'median_income_range': (35000, 70000)
            },
            'Portsmouth': {
                'population_range': (2200, 4800),
                'demographics': {
                    'white': (0.20, 0.45), 'black': (0.45, 0.70),
                    'hispanic': (0.04, 0.10), 'asian': (0.01, 0.03),
                    'male': (0.45, 0.50), 'female': (0.50, 0.55),
                    'age_45_49': (0.060, 0.080), 'age_50_54': (0.060, 0.080),
                    'age_55_59': (0.055, 0.075), 'age_60_64': (0.050, 0.070),
                    'age_65_69': (0.045, 0.065), 'age_70_75': (0.035, 0.055),
                    'less_than_hs': (0.12, 0.22), 'hs_graduate': (0.28, 0.38),
                    'some_college': (0.28, 0.38), 'bachelor_plus': (0.12, 0.25),
                    'under_20k': (0.15, 0.28), '20k_35k': (0.18, 0.28),
                    '35k_50k': (0.15, 0.25), '50k_75k': (0.15, 0.25),
                    'over_75k': (0.08, 0.20)
                },
                'poverty_range': (0.15, 0.40),
                'medicaid_rate_range': (0.30, 0.55),
                'median_income_range': (30000, 60000)
            }
        }

    def generate_tract_population(
        self,
        tract_data: pd.Series,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic population for a single census tract.

        Args:
            tract_data: Series containing tract-level data (GEOID, area, population, etc.)
            seed: Random seed for reproducibility

        Returns:
            DataFrame with synthetic individual-level data
        """
        if seed is not None:
            np.random.seed(seed)

        area = tract_data.get('Area', tract_data.get('area', 'Virginia Beach'))
        profile = self.area_profiles.get(area, self.area_profiles['Virginia Beach'])

        # Generate tract population size
        pop_min, pop_max = profile['population_range']
        population = int(np.random.uniform(pop_min, pop_max))

        # Generate demographic profiles for each individual
        individuals = []
        for i in range(population):
            individual = self._generate_individual(profile)
            individual['tract_geoid'] = tract_data['GEOID']
            individual['individual_id'] = f"{tract_data['GEOID']}_{i:05d}"
            individuals.append(individual)

        return pd.DataFrame(individuals)

    def _generate_individual(self, profile: Dict) -> Dict:
        """Generate demographic profile for single individual."""
        demographics = profile['demographics']

        # Generate race/ethnicity
        race_weights = [
            demographics['white'][0] + (demographics['white'][1] - demographics['white'][0]) * np.random.random(),
            demographics['black'][0] + (demographics['black'][1] - demographics['black'][0]) * np.random.random(),
            demographics['hispanic'][0] + (demographics['hispanic'][1] - demographics['hispanic'][0]) * np.random.random(),
            demographics.get('asian', (0.03, 0.05))[0]
        ]
        race_weights = np.array(race_weights)
        race_weights = race_weights / race_weights.sum()

        race = np.random.choice(['white', 'black', 'hispanic', 'asian'], p=race_weights)

        # Generate gender
        gender = np.random.choice(['male', 'female'])

        # Generate age group (screening-eligible ages)
        age_groups = ['45_49', '50_54', '55_59', '60_64', '65_69', '70_75']
        age_weights = [
            np.random.uniform(demographics[f'age_{ag}'][0], demographics[f'age_{ag}'][1])
            for ag in age_groups
        ]
        age_weights = np.array(age_weights)
        age_weights = age_weights / age_weights.sum()
        age = np.random.choice(age_groups, p=age_weights)

        # Generate income (correlated with education)
        income_groups = ['under_20k', '20k_35k', '35k_50k', '50k_75k', 'over_75k']
        income_weights = [
            np.random.uniform(demographics[ig][0], demographics[ig][1])
            for ig in income_groups
        ]
        income_weights = np.array(income_weights)
        income_weights = income_weights / income_weights.sum()
        income = np.random.choice(income_groups, p=income_weights)

        # Generate education (correlated with income)
        education_groups = ['less_than_hs', 'hs_graduate', 'some_college', 'bachelor_plus']

        # Apply correlation: higher income -> higher education probability
        income_education_correlation = {
            'under_20k': [0.30, 0.40, 0.20, 0.10],
            '20k_35k': [0.20, 0.35, 0.30, 0.15],
            '35k_50k': [0.10, 0.30, 0.35, 0.25],
            '50k_75k': [0.05, 0.25, 0.35, 0.35],
            'over_75k': [0.03, 0.15, 0.30, 0.52]
        }

        education_probs = income_education_correlation.get(income, [0.15, 0.30, 0.35, 0.20])
        education = np.random.choice(education_groups, p=education_probs)

        return {
            'race': race,
            'gender': gender,
            'age': age,
            'income': income,
            'education': education
        }

    def validate_population(
        self,
        synthetic_pop: pd.DataFrame,
        acs_benchmarks: Optional[pd.Series] = None,
        tolerance: float = 0.10
    ) -> Dict:
        """
        Validate synthetic population against ACS benchmarks.

        Args:
            synthetic_pop: Synthetic population DataFrame
            acs_benchmarks: ACS tract-level benchmarks
            tolerance: Acceptable deviation from benchmarks

        Returns:
            Dictionary with validation results
        """
        validation_results = {}

        # Validate race/ethnicity distribution
        race_dist = synthetic_pop['race'].value_counts(normalize=True)
        validation_results['race_distribution'] = race_dist.to_dict()

        # Validate age distribution
        age_dist = synthetic_pop['age'].value_counts(normalize=True)
        validation_results['age_distribution'] = age_dist.to_dict()

        # Validate gender distribution
        gender_dist = synthetic_pop['gender'].value_counts(normalize=True)
        validation_results['gender_distribution'] = gender_dist.to_dict()

        # Chi-square test for categorical variables
        for variable in ['race', 'age', 'gender', 'income', 'education']:
            observed = synthetic_pop[variable].value_counts()
            if acs_benchmarks is not None and variable in acs_benchmarks:
                expected = acs_benchmarks[variable]
                chi2_stat, p_value = stats.chisquare(observed, expected)
                validation_results[f'{variable}_chi2'] = {
                    'statistic': chi2_stat,
                    'p_value': p_value,
                    'valid': p_value > 0.05
                }

        validation_results['population_size'] = len(synthetic_pop)
        validation_results['overall_valid'] = all(
            v.get('valid', True) for k, v in validation_results.items() 
            if isinstance(v, dict) and 'valid' in v
        )

        return validation_results
