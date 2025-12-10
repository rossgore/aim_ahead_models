"""
Health Economics Model for Colon Cancer Screening and Treatment Costs (CSV-Driven)

Calculates expected healthcare costs for individuals based on:
- Individual cancer risk (probability of developing colorectal cancer)
- Screening status (Screened vs Not_Screened)
- Stage distribution (influenced by screening status)
- Treatment costs by cancer stage

All parameters are loaded from colon_cancer_economics_parameters.csv,
so costs can be updated without modifying code.

Screening dramatically improves prognosis:
- Screened individuals: 42% Stage I, 28% Stage II, 21% Stage III, 9% Stage IV
- Unscreened individuals: 18% Stage I, 22% Stage II, 35% Stage III, 25% Stage IV
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ColonCancerEconomicsModel:
    """
    CSV-driven health economics model for colorectal cancer screening and treatment.

    Loads all parameters from colon_cancer_economics_parameters.csv for maximum flexibility.
    Integrates individual risk, screening status, and stage-specific costs
    to calculate expected healthcare expenditures.
    """

    def __init__(self, parameters_csv: str):
        """
        Initialize the economics model from CSV parameters file.

        Args:
            parameters_csv: Path to colon_cancer_economics_parameters.csv
        """
        logger.info("Initializing Colon Cancer Economics Model (CSV-driven)...")
        
        # Load all parameters from CSV
        params_df = pd.read_csv(parameters_csv)
        
        # Parse economic parameters (cast to correct types)
        self.time_horizon = int(self._get_param_value(params_df, 'economic_parameters', 'time_horizon', 10))
        self.discount_rate = self._get_param_value(params_df, 'economic_parameters', 'discount_rate', 0.03)
        self.screening_cost = self._get_param_value(params_df, 'screening_procedures', 'colonoscopy_cost', 1500.0)
        self.advanced_stage_reduction = self._get_param_value(params_df, 'economic_parameters', 'advanced_stage_reduction', 0.30)
        self.default_cancer_risk = self._get_param_value(params_df, 'default_risks', 'default_population_cancer_risk', 0.005)
        
        # Parse treatment costs by stage
        self.treatment_costs = {}
        for stage in ['Stage_I', 'Stage_II', 'Stage_III', 'Stage_IV']:
            cost = self._get_param_value(params_df, 'treatment_costs', stage, 50000.0)
            self.treatment_costs[stage] = cost
        
        # Parse survival multipliers by stage
        self.survival_cost_multipliers = {}
        for stage in ['Stage_I', 'Stage_II', 'Stage_III', 'Stage_IV']:
            mult = self._get_param_value(params_df, 'survival_multipliers', stage, 1.0)
            self.survival_cost_multipliers[stage] = mult
        
        # Parse stage distributions
        self.stage_distributions = {
            'Screened': {},
            'Not_Screened': {}
        }
        for stage in ['Stage_I', 'Stage_II', 'Stage_III', 'Stage_IV']:
            screened_prob = self._get_param_value(params_df, 'stage_distributions_screened', stage, 0.25)
            unscreened_prob = self._get_param_value(params_df, 'stage_distributions_unscreened', stage, 0.25)
            self.stage_distributions['Screened'][stage] = screened_prob
            self.stage_distributions['Not_Screened'][stage] = unscreened_prob
        
        logger.info(f"✓ Economics Model Initialized")
        logger.info(f"  Time horizon: {self.time_horizon} years")
        logger.info(f"  Discount rate: {self.discount_rate*100}%")
        logger.info(f"  Screening cost: ${self.screening_cost:,.0f}")
        logger.info(f"  Treatment costs: Stage_I=${self.treatment_costs['Stage_I']:,.0f}, "
                   f"Stage_IV=${self.treatment_costs['Stage_IV']:,.0f}")

    def _get_param_value(self, params_df: pd.DataFrame, param_type: str, param_name: str, default: float) -> float:
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
        match = params_df[(params_df['Parameter_Type'] == param_type) & 
                          (params_df['Parameter_Name'] == param_name)]
        if len(match) > 0:
            return float(match.iloc[0]['Parameter_Value'])
        return default

    def calculate_individual_cost(self,
                                  cancer_risk: float,
                                  screening_status: str,
                                  age_group: str,
                                  is_eligible_age: bool) -> Dict[str, float]:
        """
        Calculate expected lifetime cost for an individual.

        Formula:
            Expected_Cost = Cancer_Risk × Σ(Stage_Probability × Stage_Cost × Survival_Multiplier)
                           + (Screening_Cost if Screened)
                           - (Screening_Benefit if Screened)

        Args:
            cancer_risk: Probability of developing colon cancer (0.0 to 1.0)
            screening_status: 'Screened' or 'Not_Screened'
            age_group: Age group string (e.g., '55to59')
            is_eligible_age: Boolean - is individual in screening-eligible age (45-75)

        Returns:
            Dict with cost components
        """
        # Initialize cost components
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
        stage_dist = self.stage_distributions.get(screening_status,
                                                   self.stage_distributions['Not_Screened'])
        
        expected_treatment_cost = 0.0
        for stage, prob in stage_dist.items():
            stage_cost = self.treatment_costs[stage] * self.survival_cost_multipliers[stage]
            stage_discounted_cost = stage_cost * (cancer_risk * prob)
            expected_treatment_cost += stage_discounted_cost
            cost_dict['stage_specific_costs'][stage] = stage_discounted_cost

        cost_dict['treatment_cost'] = expected_treatment_cost

        # Step 2: Add screening cost (one-time procedure)
        if screening_status == 'Screened' and is_eligible_age:
            cost_dict['screening_cost'] = self.screening_cost

            # Screening benefit: reduced risk of developing advanced cancer
            cost_dict['screening_benefit'] = (
                cancer_risk *
                (self.stage_distributions['Not_Screened']['Stage_III'] +
                 self.stage_distributions['Not_Screened']['Stage_IV']) *
                self.treatment_costs['Stage_III'] *
                self.advanced_stage_reduction
            )

        # Step 3: Calculate total cost (with discounting over time horizon)
        total_cost = cost_dict['treatment_cost'] + cost_dict['screening_cost']
        total_cost -= cost_dict['screening_benefit']

        # Apply discount rate over time horizon
        discount_factor = sum([(1 / (1 + self.discount_rate) ** year)
                              for year in range(1, int(self.time_horizon) + 1)]) / self.time_horizon
        total_cost *= discount_factor

        cost_dict['total_cost'] = max(0.0, total_cost)

        return cost_dict

    def apply_costs_to_population(self, population_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate costs for entire population.

        Expects population_df to have at least:
        - Individual_ID
        - Age_Group
        - Colon_Cancer_Screening_Status
        - Age_Eligibility
        - Unscreened_Risk (or risk columns from model.py)
        - Screened_Risk (optional)

        Args:
            population_df: DataFrame with population data

        Returns:
            DataFrame with original data plus individual cost columns
        """
        logger.info(f"Calculating costs for {len(population_df)} individuals...")
        
        costs = []

        for idx, row in population_df.iterrows():
            screening_status = row.get('Colon_Cancer_Screening_Status', 'Not_Screened')
            age_group = row.get('Age_Group', 'Unknown')
            age_eligible = row.get('Age_Eligibility', 'Outside_45_75_range') == 'Eligible_45_75'

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
                is_eligible_age=age_eligible
            )

            costs.append(cost_dict)

        # Expand cost dictionaries into columns
        costs_df = pd.json_normalize(costs)

        # Combine with original data
        result_df = population_df.copy()
        for col in costs_df.columns:
            result_df[f'Cost_{col}'] = costs_df[col]

        logger.info(f"✓ Calculated costs for {len(result_df)} individuals")
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
        report['screened_count'] = (costs_df['Cost_screening_status'] == 'Screened').sum()
        report['unscreened_count'] = (costs_df['Cost_screening_status'] == 'Not_Screened').sum()

        screened_df = costs_df[costs_df['Cost_screening_status'] == 'Screened']
        unscreened_df = costs_df[costs_df['Cost_screening_status'] == 'Not_Screened']

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

    def generate_scenario_comparison(self, baseline_costs: pd.DataFrame,
                                    policy_costs: pd.DataFrame,
                                    scenario_name: str) -> Dict:
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
            'Individual_ID', 
            'Medicaid_Status', 
            'Coverage_Status_After_Policy',
            'Cost_total_cost',
            'Cost_treatment_cost',  # This is the key metric
            'Cost_screening_status'
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

        # Individuals with changed screening status
        lost_screening = (
            (merged['Cost_screening_status_baseline'] == 'Screened') &
            (merged['Cost_screening_status_policy'] == 'Not_Screened')
        )

        # Both conditions
        lost_coverage_and_screening = lost_coverage & lost_screening

        affected_individuals = merged[lost_coverage_and_screening].copy()

        # Calculate incremental TREATMENT costs (the real burden)
        # This excludes the screening procedure cost and focuses on cancer treatment
        if len(affected_individuals) > 0:
            # Treatment cost increase = what we'll spend MORE on cancer treatment
            # because screening wasn't done
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
            
            # Key metric: Treatment cost increase (positive = bad)
            'total_treatment_cost_increase': (
                affected_individuals['treatment_cost_increase'].sum() 
                if len(affected_individuals) > 0 else 0
            ),
            'avg_treatment_cost_increase_per_affected': (
                affected_individuals['treatment_cost_increase'].mean()
                if len(affected_individuals) > 0 else 0
            ),
            
            # Also report total cost change for completeness
            'total_cost_change': (
                affected_individuals['total_cost_change'].sum()
                if len(affected_individuals) > 0 else 0
            ),
            
            'affected_individuals_dataframe': affected_individuals
        }

        return report
