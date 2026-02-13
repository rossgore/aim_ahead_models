"""
Cancer Screening Calculator - Cancer-Agnostic Version

Reusable module for cancer screening probability calculation and status assignment.
Supports both colorectal and breast cancer screening with configurable parameters.

This module encapsulates:
- Screening eligibility checks (age ranges vary by cancer type)
- Screening joint factor lookups from CSV
- Screening probability calculation with tract rates and demographic adjustments
- Screening status assignment (Screened / Not_Screened)

Usage:
    # For colon cancer
    colon_calculator = ScreeningCalculator(
        cancer_type='colon',
        screening_joint_distributions_csv='screening-joint-distributions.csv',
        screening_rates_csv='colon-rates.csv'
    )

    # For breast cancer
    breast_calculator = ScreeningCalculator(
        cancer_type='breast',
        screening_joint_distributions_csv='breast-screening-joint-distributions.csv',
        screening_rates_csv='breast-cancer-rates.csv'
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ScreeningCalculator:
    """
    Handles cancer screening probability and status assignment.

    Designed to work with:
    - Screening joint distributions CSV (tract-level adjustment factors)
    - Cancer screening rates CSV (tract-level baseline rates)

    All screening logic is INDEPENDENT of risk assessment.
    """

    # Age eligibility ranges by cancer type
    AGE_ELIGIBILITY_RANGES = {
        'colon': {
            'min_age_group': '45to49',
            'max_age_group': '75to79',
            'eligible_ages': [
                '45to49', '50to54', '55to59', '60to61', '62to64', 
                '65to66', '67to69', '70to74', '75to79'
            ],
            'display_name': 'Eligible_45-75'
        },
        'breast': {
            'min_age_group': '40to44',
            'max_age_group': '70to74',
            'eligible_ages': [
                '40to44', '45to49', '50to54', '55to59', '60to61', 
                '62to64', '65to66', '67to69', '70to74'
            ],
            'display_name': 'Eligible_40-74'
        }
    }

    def __init__(
        self, 
        cancer_type: str,
        screening_joint_distributions_csv: str, 
        screening_rates_csv: str
    ):
        """
        Initialize the screening calculator.

        Args:
            cancer_type: Type of cancer ('colon' or 'breast')
            screening_joint_distributions_csv: Path to screening adjustment factors CSV
                Expected columns: GEOID, {AGE_GROUP}_Screening_Adjustment, 
                                 {INSURANCE}_Screening_Adjustment, 
                                 {RACE}_Screening_Adjustment, etc.
            screening_rates_csv: Path to tract-level screening rates CSV
                Expected columns: GEOID, {CANCER_TYPE}_SCREEN_RATE (percentage 0-100)
        """
        if cancer_type.lower() not in ['colon', 'breast']:
            raise ValueError(f"cancer_type must be 'colon' or 'breast', got: {cancer_type}")

        self.cancer_type = cancer_type.lower()
        self.age_config = self.AGE_ELIGIBILITY_RANGES[self.cancer_type]

        logger.info(f"Initializing ScreeningCalculator for {self.cancer_type} cancer...")

        # Load screening joint distributions
        self.screening_joint_dist_df = pd.read_csv(screening_joint_distributions_csv)
        self.screening_joint_dict = dict(
            zip(self.screening_joint_dist_df['GEOID'], 
                self.screening_joint_dist_df.to_dict('records'))
        )
        logger.info(f"  Loaded screening joint distributions for "
                   f"{len(self.screening_joint_dict)} tracts")

        # Load screening rates
        screening_rates_df = pd.read_csv(screening_rates_csv)

        # Determine the screening rate column name
        if self.cancer_type == 'colon':
            rate_column = 'COLON_SCREEN_RATE'
        elif self.cancer_type == 'breast':
            rate_column = 'BREAST_SCREEN_RATE'
        else:
            rate_column = 'SCREEN_RATE'  # Fallback

        if rate_column not in screening_rates_df.columns:
            raise ValueError(
                f"Expected column '{rate_column}' not found in {screening_rates_csv}. "
                f"Available columns: {list(screening_rates_df.columns)}"
            )

        self.screening_rates_dict = dict(
            zip(screening_rates_df['GEOID'], screening_rates_df[rate_column])
        )
        logger.info(f"  Loaded screening rates for {len(self.screening_rates_dict)} tracts")
        logger.info(f"  Age eligibility: {self.age_config['display_name']}")

    def is_eligible_age_group(self, age_group: str) -> bool:
        """
        Check if age group is eligible for screening based on cancer type.

        Args:
            age_group: Age group string (e.g., '45to49', '50to54', '40to44')

        Returns:
            True if age group is in screening age range, False otherwise
        """
        return age_group in self.age_config['eligible_ages']

    def get_screening_joint_factors(
        self, 
        geoid: str, 
        age_group: str, 
        race_ethnicity: str, 
        insurance_status: str
    ) -> Tuple[float, float, float]:
        """
        Extract screening adjustment factors from screening joint distributions.

        These factors encode how screening prevalence varies by demographic groups
        relative to the tract baseline rate.

        Args:
            geoid: Census tract GEOID
            age_group: Age group (e.g., '45to49')
            race_ethnicity: Race/ethnicity category
            insurance_status: 'Insured' or 'Uninsured'

        Returns:
            Tuple of (age_adjustment, insurance_adjustment, race_adjustment)
            Each factor is a multiplier (e.g., 1.3 means 30% higher screening)
        """
        if geoid not in self.screening_joint_dict:
            # Fallback to neutral (1.0) adjustments if GEOID not found
            return (1.0, 1.0, 1.0)

        row_data = self.screening_joint_dict[geoid]

        # Age adjustment (e.g., '45to49_Screening_Adjustment')
        age_col = f"{age_group}_Screening_Adjustment"
        age_adj = row_data.get(age_col, 1.0)
        if pd.isna(age_adj):
            age_adj = 1.0

        # Insurance adjustment (e.g., 'Insured_Screening_Adjustment')
        insurance_col = f"{insurance_status}_Screening_Adjustment"
        insurance_adj = row_data.get(insurance_col, 1.0)
        if pd.isna(insurance_adj):
            insurance_adj = 1.0

        # Race adjustment (e.g., 'White_NonHispanic_Screening_Adjustment')
        race_col = f"{race_ethnicity}_Screening_Adjustment"
        race_adj = row_data.get(race_col, 1.0)
        if pd.isna(race_adj):
            race_adj = 1.0

        return (float(age_adj), float(insurance_adj), float(race_adj))

    def calculate_screening_probability(
        self, 
        tract_rate: float, 
        geoid: str, 
        age_group: str, 
        race_ethnicity: str, 
        insurance_status: str
    ) -> float:
        """
        Calculate individual screening probability using screening joint distributions.

        Formula:
            adjusted_prob = tract_rate × age_adj × insurance_adj × race_adj
            (bounded to [0.01, 0.99])

        This is INDEPENDENT of risk assessment.

        Args:
            tract_rate: Baseline tract screening rate (0.0-1.0, not percentage)
            geoid: Census tract GEOID
            age_group: Age group
            race_ethnicity: Race/ethnicity
            insurance_status: Insurance status ('Insured' or 'Uninsured')

        Returns:
            Probability of being screened (0.0-1.0)
        """
        age_adj, insurance_adj, race_adj = self.get_screening_joint_factors(
            geoid, age_group, race_ethnicity, insurance_status
        )

        combined_adjustment = age_adj * insurance_adj * race_adj
        adjusted_prob = tract_rate * combined_adjustment

        # Clip to reasonable range
        adjusted_prob = max(0.01, min(0.99, adjusted_prob))

        return adjusted_prob

    def assign_screening_to_population(
        self, 
        population_df: pd.DataFrame,
        insurance_column: str = 'Health_Insurance_Status'
    ) -> pd.DataFrame:
        """
        Assign cancer screening status to population using screening joint distributions.

        For each individual:
        1. Check if age eligible (age range varies by cancer type)
        2. If ineligible: status='Not_Screened', probability=0.0
        3. If eligible: calculate prob using tract rate + adjustments, draw Bernoulli

        Args:
            population_df: DataFrame with columns:
                - Tract_GEOID
                - Age_Group
                - Race_Ethnicity
                - {insurance_column} (default: 'Health_Insurance_Status')
            insurance_column: Name of insurance status column (allows flexibility 
                             for policy scenarios with updated insurance columns)

        Returns:
            DataFrame with new columns:
                - Age_Eligibility: 'Eligible_{age_range}' or 'Outside_{age_range}'
                - {Cancer_Type}_Screening_Probability: Calculated probability
                - {Cancer_Type}_Cancer_Screening_Status: 'Screened' or 'Not_Screened'
        """
        screening_probs = []
        screening_status = []
        age_eligibility = []

        for idx, row in population_df.iterrows():
            age_group = row['Age_Group']

            if not self.is_eligible_age_group(age_group):
                # Not eligible for screening
                screening_probs.append(0.0)
                screening_status.append('Not_Screened')
                age_eligibility.append(f"Outside_{self.age_config['display_name']}")
            else:
                # Eligible - calculate probability and assign status
                geoid = row['Tract_GEOID']

                # Get tract baseline rate (convert from percentage to decimal)
                tract_rate_pct = self.screening_rates_dict.get(geoid, 68)  # Default 68%
                tract_rate = tract_rate_pct / 100.0

                # Calculate individual probability
                prob = self.calculate_screening_probability(
                    tract_rate=tract_rate,
                    geoid=geoid,
                    age_group=age_group,
                    race_ethnicity=row['Race_Ethnicity'],
                    insurance_status=row[insurance_column]
                )

                # Draw Bernoulli to determine screening status
                screens = np.random.random() < prob

                screening_probs.append(prob)
                screening_status.append('Screened' if screens else 'Not_Screened')
                age_eligibility.append(self.age_config['display_name'])

        # Add columns to dataframe
        result = population_df.copy()
        result['Age_Eligibility'] = age_eligibility
        result[f'{self.cancer_type.capitalize()}_Screening_Probability'] = screening_probs
        result[f'{self.cancer_type.capitalize()}_Cancer_Screening_Status'] = screening_status

        return result
