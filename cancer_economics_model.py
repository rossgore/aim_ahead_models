"""
Cancer Economics Model - Cancer-Agnostic Version

Health Economics Model for Cancer Screening and Treatment Costs (CSV-Driven)

Calculates expected healthcare costs for individuals based on:
- Individual cancer risk (probability of developing cancer)
- Screening status (Screened vs Not_Screened)
- Stage distribution (influenced by screening status)
- Treatment costs by cancer stage

All parameters are loaded from cancer-specific CSV files, so costs can be 
updated without modifying code.

Supports both colorectal and breast cancer with different stage structures:
- Colon: Stage I, II, III, IV
- Breast: Stage 0 (DCIS), Stage I, II, III, IV

Usage:
    # For colon cancer
    colon_model = CancerEconomicsModel(
        cancer_type='colon',
        parameters_csv='data/colon_cancer_economics_parameters.csv'
    )

    # For breast cancer
    breast_model = CancerEconomicsModel(
        cancer_type='breast',
        parameters_csv='data/breast_cancer_economics_parameters.csv'
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CancerEconomicsModel:
    """
    CSV-driven health economics model for cancer screening and treatment.

    Loads all parameters from cancer-specific CSV files for maximum flexibility.
    Integrates individual risk, screening status, and stage-specific costs to 
    calculate expected healthcare expenditures.

    Supports multiple cancer types with different stage structures.
    """

    # Stage definitions by cancer type
    STAGE_STRUCTURES = {
        'colon': ['Stage_I', 'Stage_II', 'Stage_III', 'Stage_IV'],
        'breast': ['Stage_0', 'Stage_I', 'Stage_II', 'Stage_III', 'Stage_IV']
    }

    def __init__(self, cancer_type: str, parameters_csv: str):
        """
        Initialize the economics model from CSV parameters file.

        Args:
            cancer_type: Type of cancer ('colon' or 'breast')
            parameters_csv: Path to economics parameters CSV file
                Expected format: Parameter_Type, Parameter_Name, Parameter_Value, ...
        """
        if cancer_type.lower() not in ['colon', 'breast']:
            raise ValueError(f"cancer_type must be 'colon' or 'breast', got: {cancer_type}")

        self.cancer_type = cancer_type.lower()
        self.stages = self.STAGE_STRUCTURES[self.cancer_type]

        logger.info(f"Initializing {self.cancer_type.capitalize()} Cancer Economics Model "
                   f"(CSV-driven)...")

        # Load all parameters from CSV
        params_df = pd.read_csv(parameters_csv)

        # Economic parameters
        self.time_horizon = int(self._get_param_value(
            params_df, 'economic_parameters', 'time_horizon', 10
        ))
        self.discount_rate = self._get_param_value(
            params_df, 'economic_parameters', 'discount_rate', 0.03
        )
        self.advanced_stage_reduction = self._get_param_value(
            params_df, 'economic_parameters', 'advanced_stage_reduction', 0.30
        )

        # Screening cost (colonoscopy vs mammography)
        if self.cancer_type == 'colon':
            self.screening_cost = self._get_param_value(
                params_df, 'screening_procedures', 'colonoscopy_cost', 1500.0
            )
        elif self.cancer_type == 'breast':
            self.screening_cost = self._get_param_value(
                params_df, 'screening_procedures', 'mammography_cost', 200.0
            )

        # Default risk
        self.default_cancer_risk = self._get_param_value(
            params_df, 'default_risks', 'default_population_cancer_risk', 0.005
        )

        # Parse treatment costs by stage
        self.treatment_costs = {}
        for stage in self.stages:
            cost = self._get_param_value(params_df, 'treatment_costs', stage, 50000.0)
            self.treatment_costs[stage] = cost

        # Parse survival cost multipliers by stage
        self.survival_cost_multipliers = {}
        for stage in self.stages:
            mult = self._get_param_value(params_df, 'survival_multipliers', stage, 1.0)
            self.survival_cost_multipliers[stage] = mult

        # Parse stage distributions
        self.stage_distributions = {
            'Screened': {},
            'Not_Screened': {}
        }

        for stage in self.stages:
            screened_prob = self._get_param_value(
                params_df, 'stage_distributions_screened', stage, 0.25
            )
            unscreened_prob = self._get_param_value(
                params_df, 'stage_distributions_unscreened', stage, 0.25
            )
            self.stage_distributions['Screened'][stage] = screened_prob
            self.stage_distributions['Not_Screened'][stage] = unscreened_prob

        logger.info(f"  Economics Model Initialized")
        logger.info(f"  Cancer type: {self.cancer_type}")
        logger.info(f"  Stages: {', '.join(self.stages)}")
        logger.info(f"  Time horizon: {self.time_horizon} years")
        logger.info(f"  Discount rate: {self.discount_rate*100:.1f}%")
        logger.info(f"  Screening cost: ${self.screening_cost:,.0f}")
        logger.info(f"  Treatment costs: ${self.treatment_costs[self.stages[0]]:,.0f} "
                   f"to ${self.treatment_costs[self.stages[-1]]:,.0f}")

    def _get_param_value(
        self, 
        params_df: pd.DataFrame, 
        param_type: str, 
        param_name: str, 
        default: float
    ) -> float:
        """
        Extract parameter value from DataFrame.

        Args:
            params_df: Parameters dataframe
            param_type: Parameter_Type (e.g., 'economic_parameters')
            param_name: Parameter_Name (e.g., 'discount_rate')
            default: Default value if not found

        Returns:
            Parameter value or default
        """
        match = params_df[
            (params_df['Parameter_Type'] == param_type) & 
            (params_df['Parameter_Name'] == param_name)
        ]
        if len(match) > 0:
            return float(match.iloc[0]['Parameter_Value'])
        return default

    def calculate_individual_cost(
        self, 
        cancer_risk: float, 
        screening_status: str, 
        age_group: str, 
        is_eligible_age: bool
    ) -> Dict[str, float]:
        """
        Calculate expected lifetime cost for an individual.

        Formula:
            Expected_Cost = Cancer_Risk × Σ(Stage_Probability × Stage_Cost × Survival_Multiplier)
                          + Screening_Cost (if Screened)
                          - Screening_Benefit (if Screened)

        Args:
            cancer_risk: Probability of developing cancer (0.0 to 1.0)
            screening_status: 'Screened' or 'Not_Screened'
            age_group: Age group string (e.g., '55to59')
            is_eligible_age: Boolean - is individual in screening-eligible age

        Returns:
            Dict with cost components
        """
        cost_dict = {
            'cancer_risk': cancer_risk,
            'screening_status': screening_status,
            'age_group': age_group,
            'is_eligible_age': is_eligible_age,
            'stage_specific_costs': {},
            'treatment_cost': 0.0,
            'screening_cost': 0.0,
            'screening_benefit': 0.0,
            'total_cost': 0.0
        }

        # If individual is outside screening-eligible age, no screening cost
        if not is_eligible_age:
            screening_status = 'Not_Screened'

        # Step 1: Calculate treatment cost based on stage distribution
        stage_dist = self.stage_distributions.get(
            screening_status, 
            self.stage_distributions['Not_Screened']
        )

        expected_treatment_cost = 0.0
        for stage, prob in stage_dist.items():
            stage_cost = (self.treatment_costs[stage] * 
                         self.survival_cost_multipliers[stage])
            stage_discounted_cost = stage_cost * cancer_risk * prob
            expected_treatment_cost += stage_discounted_cost
            cost_dict['stage_specific_costs'][stage] = stage_discounted_cost

        cost_dict['treatment_cost'] = expected_treatment_cost

        # Step 2: Add screening cost (one-time procedure if screened and eligible)
        if screening_status == 'Screened' and is_eligible_age:
            cost_dict['screening_cost'] = self.screening_cost

        # Step 3: Calculate screening benefit (reduced advanced-stage cancer costs)
        if screening_status == 'Screened' and is_eligible_age:
            # Benefit = reduced risk of advanced stages × advanced stage costs
            advanced_stages = self.stages[-2:]  # Last 2 stages (III, IV)
            advanced_stage_prob = sum(
                self.stage_distributions['Not_Screened'].get(s, 0) 
                for s in advanced_stages
            )
            avg_advanced_cost = np.mean([
                self.treatment_costs[s] for s in advanced_stages
            ])
            cost_dict['screening_benefit'] = (
                cancer_risk * advanced_stage_prob * avg_advanced_cost * 
                self.advanced_stage_reduction
            )

        # Step 4: Calculate total cost with discounting over time horizon
        total_cost = (cost_dict['treatment_cost'] + 
                     cost_dict['screening_cost'] - 
                     cost_dict['screening_benefit'])

        # Apply discount factor over time horizon
        discount_factor = sum(
            1 / (1 + self.discount_rate) ** year 
            for year in range(1, int(self.time_horizon) + 1)
        ) / self.time_horizon

        total_cost *= discount_factor
        cost_dict['total_cost'] = max(0.0, total_cost)

        return cost_dict

    def apply_costs_to_population(
        self, 
        population_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate costs for entire population.

        Expects population_df to have at least:
        - Individual_ID
        - Age_Group
        - {Cancer_Type}_Cancer_Screening_Status (e.g., 'Colon_Cancer_Screening_Status')
        - Age_Eligibility
        - Unscreened_Risk or risk columns from model.py
        - Screened_Risk (optional)

        Args:
            population_df: DataFrame with population data

        Returns:
            DataFrame with original data plus individual cost columns
        """
        logger.info(f"Calculating costs for {len(population_df)} individuals...")

        # Determine column names based on cancer type
        screening_status_col = f'{self.cancer_type.capitalize()}_Cancer_Screening_Status'

        # Fallback for backward compatibility
        if screening_status_col not in population_df.columns:
            # Try old naming convention
            if 'Colon_Cancer_Screening_Status' in population_df.columns:
                screening_status_col = 'Colon_Cancer_Screening_Status'
            else:
                raise ValueError(
                    f"Required column '{screening_status_col}' not found in population_df. "
                    f"Available columns: {list(population_df.columns)}"
                )

        costs = []

        for idx, row in population_df.iterrows():
            screening_status = row.get(screening_status_col, 'Not_Screened')
            age_group = row.get('Age_Group', 'Unknown')
            age_eligible = row.get('Age_Eligibility', 'Outside_45-75_range')

            # Check if eligible (contains "Eligible" in the string)
            is_eligible = 'Eligible' in str(age_eligible)

            # Determine cancer risk
            if pd.isna(row.get('Unscreened_Risk')) or row.get('Risk_Category') == 'Outside Screening Age':
                cancer_risk = self.default_cancer_risk
            else:
                if screening_status == 'Screened' and not pd.isna(row.get('Screened_Risk')):
                    cancer_risk = row.get('Screened_Risk', self.default_cancer_risk) / 100.0
                else:
                    cancer_risk = row.get('Unscreened_Risk', self.default_cancer_risk) / 100.0

            # Calculate costs for this individual
            cost_dict = self.calculate_individual_cost(
                cancer_risk=cancer_risk,
                screening_status=screening_status,
                age_group=age_group,
                is_eligible_age=is_eligible
            )
            costs.append(cost_dict)

        # Expand cost dictionaries into columns
        costs_df = pd.json_normalize(costs)

        # Combine with original data
        result_df = population_df.copy()
        for col in costs_df.columns:
            result_df[f'Cost_{col}'] = costs_df[col]

        logger.info(f"  Calculated costs for {len(result_df)} individuals")
        return result_df

    def generate_cost_report(self, costs_df: pd.DataFrame) -> Dict:
        """
        Generate summary report of population costs.

        Args:
            costs_df: DataFrame with costs calculated

        Returns:
            Dictionary with summary statistics
        """
        report = {
            'total_population': len(costs_df),
            'total_cost': costs_df['Cost_total_cost'].sum(),
            'avg_cost_per_person': costs_df['Cost_total_cost'].mean(),
            'median_cost_per_person': costs_df['Cost_total_cost'].median(),
            'min_cost': costs_df['Cost_total_cost'].min(),
            'max_cost': costs_df['Cost_total_cost'].max(),
        }

        # Breakdown by screening status
        screening_status_col = f'Cost_screening_status'
        report['screened_count'] = (costs_df[screening_status_col] == 'Screened').sum()
        report['unscreened_count'] = (costs_df[screening_status_col] == 'Not_Screened').sum()

        screened_df = costs_df[costs_df[screening_status_col] == 'Screened']
        unscreened_df = costs_df[costs_df[screening_status_col] == 'Not_Screened']

        if len(screened_df) > 0:
            report['avg_cost_screened'] = screened_df['Cost_total_cost'].mean()
            report['total_cost_screened'] = screened_df['Cost_total_cost'].sum()

        if len(unscreened_df) > 0:
            report['avg_cost_unscreened'] = unscreened_df['Cost_total_cost'].mean()
            report['total_cost_unscreened'] = unscreened_df['Cost_total_cost'].sum()

        # Cost difference
        if len(screened_df) > 0 and len(unscreened_df) > 0:
            report['cost_difference_screening_vs_unscreened'] = (
                report['avg_cost_screened'] - report['avg_cost_unscreened']
            )
            report['net_savings_from_screening'] = (
                report['total_cost_unscreened'] - report['total_cost_screened']
            )

        return report

    def generate_scenario_comparison(
        self, 
        baseline_costs: pd.DataFrame, 
        policy_costs: pd.DataFrame, 
        scenario_name: str
    ) -> Dict:
        """
        Compare costs between baseline and policy scenario.

        Focuses on individuals who lost Medicaid and changed screening status.

        KEY METRIC: Treatment cost increase due to missed screening 
        (excludes screening procedure cost to focus on cancer treatment burden)

        Args:
            baseline_costs: DataFrame with costs from baseline scenario
            policy_costs: DataFrame with costs from policy scenario
            scenario_name: Name of policy scenario for reporting

        Returns:
            Dictionary with comparison metrics
        """
        # Select columns to merge from policy_costs
        policy_merge_cols = [
            'Individual_ID', 'Medicaid_Status', 'Coverage_Status_After_Policy',
            'Cost_total_cost', 'Cost_treatment_cost', 'Cost_screening_status'
        ]
        policy_cols_to_merge = [col for col in policy_merge_cols if col in policy_costs.columns]

        # Merge on Individual_ID to track changes
        merged = baseline_costs.merge(
            policy_costs[policy_cols_to_merge],
            on='Individual_ID',
            suffixes=('_baseline', '_policy'),
            how='left'
        )

        # Identify individuals who lost coverage
        lost_coverage = (
            (merged['Medicaid_Status_baseline'] == True) & 
            (merged['Coverage_Status_After_Policy_policy'] == 'Uninsured')
        )

        # Identify individuals with changed screening status
        lost_screening = (
            (merged['Cost_screening_status_baseline'] == 'Screened') & 
            (merged['Cost_screening_status_policy'] == 'Not_Screened')
        )

        # Both conditions
        lost_coverage_and_screening = lost_coverage & lost_screening
        affected_individuals = merged[lost_coverage_and_screening].copy()

        if len(affected_individuals) > 0:
            # Calculate treatment cost increase (this is the key metric)
            # This excludes the screening procedure cost and focuses on cancer treatment
            affected_individuals['treatment_cost_increase'] = (
                affected_individuals['Cost_treatment_cost_policy'] - 
                affected_individuals['Cost_treatment_cost_baseline']
            )

            # Also calculate total cost change for reference
            affected_individuals['total_cost_change'] = (
                affected_individuals['Cost_total_cost_policy'] - 
                affected_individuals['Cost_total_cost_baseline']
            )

        report = {
            'scenario_name': scenario_name,
            'total_individuals_analyzed': len(merged),
            'individuals_lost_coverage': lost_coverage.sum(),
            'individuals_lost_screening': lost_screening.sum(),
            'individuals_lost_both': lost_coverage_and_screening.sum(),
            'total_treatment_cost_increase': (
                affected_individuals['treatment_cost_increase'].sum() 
                if len(affected_individuals) > 0 else 0
            ),
            'avg_treatment_cost_increase_per_affected': (
                affected_individuals['treatment_cost_increase'].mean() 
                if len(affected_individuals) > 0 else 0
            ),
            'total_cost_change': (
                affected_individuals['total_cost_change'].sum() 
                if len(affected_individuals) > 0 else 0
            ),
            'affected_individuals_dataframe': affected_individuals
        }

        return report
