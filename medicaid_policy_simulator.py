"""
MEDICAID POLICY SIMULATOR - CANCER-AGNOSTIC VERSION

Simulates the impact of Medicaid eligibility policy changes on cancer screening
rates and healthcare costs.

Supports both colorectal and breast cancer with appropriate screening models
and economic parameters.

Key Features:
- Cancer-agnostic design (supports 'colon' and 'breast')
- Uses refactored ScreeningCalculator for screening reassignment
- Uses refactored CancerEconomicsModel for cost calculations
- Automatic parameter file detection based on cancer type
- Multiple policy scenarios (income tightening, asset tests, work requirements)

Usage:
    # Colon cancer policy simulation
    python medicaid_policy_simulator.py \
        --cancer-type colon \
        --baseline-population data/synthetic_population_colon.csv \
        --screening-joint-dist data/colon-screening-joint-distributions.csv \
        --screening-rates data/colon-rates.csv \
        --economics-parameters data/colon_cancer_economics_parameters.csv \
        --policy-scenario income_tightening \
        --output-dir output/policy_colon/

    # Breast cancer policy simulation
    python medicaid_policy_simulator.py \
        --cancer-type breast \
        --baseline-population data/synthetic_population_breast.csv \
        --screening-joint-dist data/breast-screening-joint-distributions.csv \
        --screening-rates data/breast-cancer-rates.csv \
        --economics-parameters data/breast_cancer_economics_parameters.csv \
        --policy-scenario income_tightening \
        --output-dir output/policy_breast/
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Optional
import argparse

from screening_calculator import ScreeningCalculator
from cancer_economics_model import CancerEconomicsModel

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class MedicaidPolicySimulator:
    """
    Simulate Medicaid policy changes and their impact on cancer screening and costs.

    Supports both colon and breast cancer with appropriate models.
    """

    # Medicaid income eligibility thresholds by state
    MEDICAID_THRESHOLDS = {
        'virginia': {
            'current_fpl_percent': 138,  # Current Medicaid expansion (138% FPL)
            'baseline_income_threshold': 'Less10k,10to15k,15to20k,20to25k,25to30k',
            'tightened_income_threshold': 'Less10k,10to15k,15to20k'
        }
    }

    def __init__(
        self,
        cancer_type: str,
        baseline_population_csv: str,
        screening_joint_distributions_csv: str,
        screening_rates_csv: str,
        economics_parameters_csv: str,
        state: str = 'virginia'
    ):
        """
        Initialize the policy simulator.

        Args:
            cancer_type: 'colon' or 'breast'
            baseline_population_csv: Synthetic population from model.py
            screening_joint_distributions_csv: Screening adjustment factors
            screening_rates_csv: Tract-level screening rates
            economics_parameters_csv: Economic parameters for cost calculation
            state: State for Medicaid thresholds (default: 'virginia')
        """
        if cancer_type.lower() not in ['colon', 'breast']:
            raise ValueError(f"cancer_type must be 'colon' or 'breast', got: {cancer_type}")

        self.cancer_type = cancer_type.lower()
        self.state = state.lower()

        logger.info("=" * 80)
        logger.info(f"INITIALIZING MEDICAID POLICY SIMULATOR")
        logger.info(f"Cancer Type: {self.cancer_type.upper()}")
        logger.info(f"State: {self.state.upper()}")
        logger.info("=" * 80)

        # Load baseline population
        logger.info(f"1. Loading baseline population from {baseline_population_csv}")
        self.baseline_population = pd.read_csv(baseline_population_csv)
        logger.info(f"   Loaded {len(self.baseline_population)} individuals")

        # Initialize screening calculator
        logger.info(f"2. Initializing Screening Calculator ({self.cancer_type} cancer)...")
        self.screening_calculator = ScreeningCalculator(
            cancer_type=self.cancer_type,
            screening_joint_distributions_csv=screening_joint_distributions_csv,
            screening_rates_csv=screening_rates_csv
        )

        # Initialize economics model
        logger.info(f"3. Initializing Economics Model ({self.cancer_type} cancer)...")
        self.economics_model = CancerEconomicsModel(
            cancer_type=self.cancer_type,
            parameters_csv=economics_parameters_csv
        )

        # Identify baseline Medicaid population
        self._identify_baseline_medicaid()

        logger.info("✓ Simulator initialized successfully")
        logger.info("=" * 80)

    def _identify_baseline_medicaid(self):
        """Identify who is on Medicaid in baseline scenario."""
        thresholds = self.MEDICAID_THRESHOLDS[self.state]
        eligible_incomes = thresholds['baseline_income_threshold'].split(',')

        # Medicaid eligibility: Low income + currently insured
        # (We assume currently insured low-income individuals have Medicaid)
        self.baseline_population['Medicaid_Status'] = (
            (self.baseline_population['Income_Bracket'].isin(eligible_incomes)) &
            (self.baseline_population['Health_Insurance_Status'] == 'Insured')
        )

        medicaid_count = self.baseline_population['Medicaid_Status'].sum()
        medicaid_pct = (medicaid_count / len(self.baseline_population)) * 100

        logger.info(f"\nBaseline Medicaid Status:")
        logger.info(f"  Medicaid enrollees: {medicaid_count} ({medicaid_pct:.1f}%)")
        logger.info(f"  Non-Medicaid: {len(self.baseline_population) - medicaid_count} "
                   f"({100 - medicaid_pct:.1f}%)")

    def simulate_income_tightening(
        self,
        new_threshold_brackets: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Simulate tightening Medicaid income eligibility.

        Args:
            new_threshold_brackets: Comma-separated income brackets (default: use state config)

        Returns:
            DataFrame with policy scenario applied
        """
        logger.info("=" * 80)
        logger.info("POLICY SCENARIO: INCOME ELIGIBILITY TIGHTENING")
        logger.info("=" * 80)

        if new_threshold_brackets is None:
            thresholds = self.MEDICAID_THRESHOLDS[self.state]
            new_threshold_brackets = thresholds['tightened_income_threshold']

        new_eligible_incomes = new_threshold_brackets.split(',')

        logger.info(f"\nPolicy details:")
        logger.info(f"  Current threshold: 138% FPL (~$20,000-30,000 for individual)")
        logger.info(f"  New threshold: 100% FPL (~$15,000 for individual)")
        logger.info(f"  Eligible income brackets: {new_eligible_incomes}")

        # Create policy scenario population
        policy_pop = self.baseline_population.copy()

        # Determine who loses Medicaid
        policy_pop['Still_Eligible'] = (
            policy_pop['Income_Bracket'].isin(new_eligible_incomes)
        )

        policy_pop['Lost_Medicaid'] = (
            (policy_pop['Medicaid_Status'] == True) &
            (policy_pop['Still_Eligible'] == False)
        )

        # Update insurance status for those who lost Medicaid
        policy_pop.loc[policy_pop['Lost_Medicaid'], 'Health_Insurance_Status'] = 'Uninsured'
        policy_pop['Coverage_Status_After_Policy'] = policy_pop['Health_Insurance_Status']

        # Report impact
        lost_count = policy_pop['Lost_Medicaid'].sum()
        lost_pct = (lost_count / policy_pop['Medicaid_Status'].sum()) * 100 if policy_pop['Medicaid_Status'].sum() > 0 else 0

        logger.info(f"\nImpact:")
        logger.info(f"  Individuals losing Medicaid: {lost_count} ({lost_pct:.1f}% of Medicaid population)")
        logger.info(f"  Now uninsured: {(policy_pop['Coverage_Status_After_Policy'] == 'Uninsured').sum()}")

        return policy_pop

    def simulate_asset_test(
        self,
        asset_limit: float = 2000
    ) -> pd.DataFrame:
        """
        Simulate adding asset test to Medicaid eligibility.

        Args:
            asset_limit: Maximum allowable assets (default: $2,000)

        Returns:
            DataFrame with policy scenario applied
        """
        logger.info("=" * 80)
        logger.info("POLICY SCENARIO: ASSET TEST REQUIREMENT")
        logger.info("=" * 80)

        logger.info(f"\nPolicy details:")
        logger.info(f"  Asset limit: ${asset_limit:,.0f}")
        logger.info(f"  Estimation: Individuals with income >$50k likely exceed asset limit")

        # Create policy scenario population
        policy_pop = self.baseline_population.copy()

        # Proxy for assets: Higher income brackets likely have savings
        high_asset_brackets = ['60to75k', '75to100k', '100to125k', '125to150k', '150to200k', '200kplus']

        policy_pop['Exceeds_Asset_Limit'] = (
            policy_pop['Income_Bracket'].isin(high_asset_brackets)
        )

        policy_pop['Lost_Medicaid'] = (
            (policy_pop['Medicaid_Status'] == True) &
            (policy_pop['Exceeds_Asset_Limit'] == True)
        )

        # Update insurance status
        policy_pop.loc[policy_pop['Lost_Medicaid'], 'Health_Insurance_Status'] = 'Uninsured'
        policy_pop['Coverage_Status_After_Policy'] = policy_pop['Health_Insurance_Status']

        # Report impact
        lost_count = policy_pop['Lost_Medicaid'].sum()

        logger.info(f"\nImpact:")
        logger.info(f"  Individuals losing Medicaid: {lost_count}")
        logger.info(f"  (Note: This is a proxy - actual impact depends on real asset data)")

        return policy_pop

    def simulate_work_requirement(
        self,
        required_hours_per_week: int = 20
    ) -> pd.DataFrame:
        """
        Simulate adding work requirement to Medicaid eligibility.

        Args:
            required_hours_per_week: Required work hours (default: 20)

        Returns:
            DataFrame with policy scenario applied
        """
        logger.info("=" * 80)
        logger.info("POLICY SCENARIO: WORK REQUIREMENT")
        logger.info("=" * 80)

        logger.info(f"\nPolicy details:")
        logger.info(f"  Required work hours: {required_hours_per_week} hours/week")
        logger.info(f"  Estimation: 15% of Medicaid population unable to meet requirement")

        # Create policy scenario population
        policy_pop = self.baseline_population.copy()

        # Estimate: 15% of Medicaid population cannot meet work requirement
        # (elderly, disabled, caregivers, etc.)
        medicaid_pop = policy_pop[policy_pop['Medicaid_Status'] == True].copy()

        if len(medicaid_pop) > 0:
            unable_to_work_count = int(len(medicaid_pop) * 0.15)
            unable_indices = np.random.choice(
                medicaid_pop.index,
                size=unable_to_work_count,
                replace=False
            )

            policy_pop['Unable_To_Meet_Work_Req'] = False
            policy_pop.loc[unable_indices, 'Unable_To_Meet_Work_Req'] = True

            policy_pop['Lost_Medicaid'] = (
                (policy_pop['Medicaid_Status'] == True) &
                (policy_pop['Unable_To_Meet_Work_Req'] == True)
            )

            # Update insurance status
            policy_pop.loc[policy_pop['Lost_Medicaid'], 'Health_Insurance_Status'] = 'Uninsured'
            policy_pop['Coverage_Status_After_Policy'] = policy_pop['Health_Insurance_Status']

            lost_count = policy_pop['Lost_Medicaid'].sum()
            lost_pct = (lost_count / len(medicaid_pop)) * 100

            logger.info(f"\nImpact:")
            logger.info(f"  Individuals losing Medicaid: {lost_count} ({lost_pct:.1f}% of Medicaid population)")
        else:
            policy_pop['Lost_Medicaid'] = False
            policy_pop['Coverage_Status_After_Policy'] = policy_pop['Health_Insurance_Status']
            logger.info(f"\nNo Medicaid population found - no impact")

        return policy_pop

    def reassign_screening(self, policy_population: pd.DataFrame) -> pd.DataFrame:
        """
        Reassign cancer screening status after policy change.

        Uses ScreeningCalculator with updated insurance status.

        Args:
            policy_population: Population with updated insurance status

        Returns:
            DataFrame with updated screening status
        """
        logger.info("=" * 80)
        logger.info(f"REASSIGNING {self.cancer_type.upper()} CANCER SCREENING STATUS")
        logger.info("=" * 80)

        # Use the updated insurance column
        with_new_screening = self.screening_calculator.assign_screening_to_population(
            policy_population,
            insurance_column='Coverage_Status_After_Policy'
        )
        
        # Duplicate screening columns to indicate policy scenario 
        # (We duplicate instead of renaming so the economics model can still find the standard column)
        screening_status_col = f'{self.cancer_type.capitalize()}_Cancer_Screening_Status'
        screening_prob_col = f'{self.cancer_type.capitalize()}_Screening_Probability'

        with_new_screening[f'{screening_status_col}_After_Policy'] = with_new_screening[screening_status_col]
        with_new_screening[f'{screening_prob_col}_After_Policy'] = with_new_screening[screening_prob_col]

        # Report screening changes
        lost_medicaid = with_new_screening['Lost_Medicaid'] == True

        if lost_medicaid.sum() > 0:
            baseline_screening_col = screening_status_col  # Original column from baseline
            if baseline_screening_col in with_new_screening.columns:
                baseline_screened = (
                    with_new_screening.loc[lost_medicaid, baseline_screening_col] == 'Screened'
                ).sum()
            else:
                baseline_screened = 0

            policy_screened = (
                with_new_screening.loc[lost_medicaid, f'{screening_status_col}_After_Policy'] == 'Screened'
            ).sum()

            lost_screening = baseline_screened - policy_screened

            logger.info(f"\nScreening Impact (among those who lost Medicaid):")
            logger.info(f"  Baseline screened: {baseline_screened}")
            logger.info(f"  After policy screened: {policy_screened}")
            logger.info(f"  Lost screening: {lost_screening}")

        return with_new_screening

    def calculate_costs(
        self,
        baseline_population: pd.DataFrame,
        policy_population: pd.DataFrame
    ) -> Dict:
        """
        Calculate healthcare costs for baseline and policy scenarios.

        Args:
            baseline_population: Baseline population with costs
            policy_population: Policy population with updated screening

        Returns:
            Dictionary with cost comparison
        """
        logger.info("=" * 80)
        logger.info("CALCULATING HEALTHCARE COSTS")
        logger.info("=" * 80)

        # --- FIX: Align Baseline Columns ---
        # Ensure baseline has the policy-specific columns with default values 
        # so Pandas properly appends '_policy' suffixes during the economics model merge.
        baseline_copy = baseline_population.copy()
        
        baseline_copy['Coverage_Status_After_Policy'] = baseline_copy['Health_Insurance_Status']
        if 'Lost_Medicaid' not in baseline_copy.columns:
            baseline_copy['Lost_Medicaid'] = False
        if 'Still_Eligible' not in baseline_copy.columns:
            baseline_copy['Still_Eligible'] = True
            
        screening_status_col = f'{self.cancer_type.capitalize()}_Cancer_Screening_Status'
        screening_prob_col = f'{self.cancer_type.capitalize()}_Screening_Probability'
        
        if screening_status_col in baseline_copy.columns:
            baseline_copy[f'{screening_status_col}_After_Policy'] = baseline_copy[screening_status_col]
        if screening_prob_col in baseline_copy.columns:
            baseline_copy[f'{screening_prob_col}_After_Policy'] = baseline_copy[screening_prob_col]
        # -----------------------------------

        # Calculate baseline costs (using the aligned copy)
        logger.info("\n1. Calculating baseline costs...")
        baseline_with_costs = self.economics_model.apply_costs_to_population(
            baseline_copy
        )

        # Calculate policy costs
        logger.info("\n2. Calculating policy scenario costs...")
        policy_with_costs = self.economics_model.apply_costs_to_population(
            policy_population
        )

        # Generate comparison
        logger.info("\n3. Generating cost comparison...")
        comparison = self.economics_model.generate_scenario_comparison(
            baseline_costs=baseline_with_costs,
            policy_costs=policy_with_costs,
            scenario_name="Policy Scenario"
        )

        return {
            'baseline_costs': baseline_with_costs,
            'policy_costs': policy_with_costs,
            'comparison': comparison
        }
    def run_scenario(
        self,
        scenario: str,
        output_dir: str,
        **scenario_kwargs
    ) -> Dict:
        """
        Run complete policy scenario simulation.

        Args:
            scenario: Policy scenario name ('income_tightening', 'asset_test', 'work_requirement')
            output_dir: Directory for output files
            **scenario_kwargs: Additional arguments for specific scenarios

        Returns:
            Dictionary with simulation results
        """
        logger.info("=" * 80)
        logger.info(f"RUNNING POLICY SCENARIO: {scenario.upper()}")
        logger.info(f"Cancer Type: {self.cancer_type.upper()}")
        logger.info("=" * 80)

        # Apply policy scenario
        if scenario == 'income_tightening':
            policy_pop = self.simulate_income_tightening(**scenario_kwargs)
        elif scenario == 'asset_test':
            policy_pop = self.simulate_asset_test(**scenario_kwargs)
        elif scenario == 'work_requirement':
            policy_pop = self.simulate_work_requirement(**scenario_kwargs)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        # Reassign screening
        policy_pop_with_screening = self.reassign_screening(policy_pop)

        # Calculate costs
        cost_results = self.calculate_costs(
            baseline_population=self.baseline_population,
            policy_population=policy_pop_with_screening
        )

        # Save outputs
        os.makedirs(output_dir, exist_ok=True)

        baseline_output = os.path.join(output_dir, f'baseline_population_{self.cancer_type}.csv')
        policy_output = os.path.join(output_dir, f'policy_population_{self.cancer_type}_{scenario}.csv')
        comparison_output = os.path.join(output_dir, f'cost_comparison_{self.cancer_type}_{scenario}.csv')

        cost_results['baseline_costs'].to_csv(baseline_output, index=False)
        cost_results['policy_costs'].to_csv(policy_output, index=False)

        # Save comparison summary
        comparison_summary = pd.DataFrame([cost_results['comparison']])
        comparison_summary.to_csv(comparison_output, index=False)

        logger.info("=" * 80)
        logger.info("SIMULATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nOutput files saved to: {output_dir}")
        logger.info(f"  - {baseline_output}")
        logger.info(f"  - {policy_output}")
        logger.info(f"  - {comparison_output}")

        self._print_summary(cost_results['comparison'])

        return {
            'scenario': scenario,
            'policy_population': policy_pop_with_screening,
            'cost_results': cost_results,
            'output_dir': output_dir
        }

    def _print_summary(self, comparison: Dict):
        """Print summary of policy impact."""
        logger.info("=" * 80)
        logger.info("POLICY IMPACT SUMMARY")
        logger.info("=" * 80)

        logger.info(f"\nPopulation Impact:")
        logger.info(f"  Total individuals analyzed: {comparison['total_individuals_analyzed']:,}")
        logger.info(f"  Lost coverage: {comparison['individuals_lost_coverage']:,}")
        logger.info(f"  Lost screening: {comparison['individuals_lost_screening']:,}")
        logger.info(f"  Lost both: {comparison['individuals_lost_both']:,}")

        logger.info(f"\nEconomic Impact:")
        logger.info(f"  Total treatment cost increase: ${comparison['total_treatment_cost_increase']:,.2f}")
        if comparison['individuals_lost_both'] > 0:
            logger.info(f"  Avg increase per affected individual: "
                       f"${comparison['avg_treatment_cost_increase_per_affected']:,.2f}")

        logger.info(f"\nInterpretation:")
        logger.info(f"  • Treatment costs increase when screening is lost")
        logger.info(f"  • Later-stage diagnosis → more expensive treatment")
        logger.info(f"  • Savings from avoided screening < costs from late detection")

        logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Medicaid Policy Simulator (Cancer-Agnostic)'
    )
    parser.add_argument('--cancer-type', required=True, choices=['colon', 'breast'],
                       help='Type of cancer (colon or breast)')
    parser.add_argument('--baseline-population', required=True,
                       help='Baseline synthetic population CSV')
    parser.add_argument('--screening-joint-dist', required=True,
                       help='Screening joint distributions CSV')
    parser.add_argument('--screening-rates', required=True,
                       help='Cancer screening rates CSV')
    parser.add_argument('--economics-parameters', required=True,
                       help='Economics parameters CSV')
    parser.add_argument('--policy-scenario', required=True,
                       choices=['income_tightening', 'asset_test', 'work_requirement'],
                       help='Policy scenario to simulate')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--state', default='virginia',
                       help='State for Medicaid thresholds (default: virginia)')

    args = parser.parse_args()

    simulator = MedicaidPolicySimulator(
        cancer_type=args.cancer_type,
        baseline_population_csv=args.baseline_population,
        screening_joint_distributions_csv=args.screening_joint_dist,
        screening_rates_csv=args.screening_rates,
        economics_parameters_csv=args.economics_parameters,
        state=args.state
    )

    results = simulator.run_scenario(
        scenario=args.policy_scenario,
        output_dir=args.output_dir
    )
