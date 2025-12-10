"""
Medicaid Policy Simulation Engine (CSV-Driven + Screening Recalculation)

A fully configurable Medicaid coverage policy simulator that reads policy rules
from external CSV files instead of hard-coding policy logic.

After applying coverage changes, it can recompute colon cancer screening
status using the same screening logic as model.py (via ScreeningCalculator).

Policy CSV Schema:
──────────────────
Each policy is defined in a CSV file with columns:
  - Policy_Name: Human-readable name/identifier
  - Target_Field: Field to check (e.g., 'Medicaid_Status')
  - Condition: Comparison operator (e.g., '==')
  - Value: Target value (e.g., 'True')
  - Action: Named condition handler (e.g., 'Income_Between_100_138_FPL')
  - Coverage_Change: Resulting coverage status ('Uninsured', 'Keep', etc.)
  - Note: Explanation of the policy rule

Multiple rows in a single policy CSV are OR-ed: if an individual matches ANY
row's combined condition, the first matching rule applies.

Named Actions (built-in condition handlers):
────────────────────────────────────────────
  - Income_Between_100_138_FPL: Medicaid enrollees earning $25.7k–$35.6k
  - Age_18_55_Random_<rate>: Working-age Medicaid enrollees, <rate> churn rate
  - Immigrant_Proxy: Medicaid enrollees estimated as immigrants by race/ethnicity

Example Usage:
──────────────
  python3 medicaid_policy_simulator.py \
    --population output/synthetic_population.csv \
    --screening-joint-dist data/screening-joint-distributions.csv \
    --colon-rates data/colon-rates.csv \
    --policy policies/income_tightening.csv \
    --policy policies/admin_churn.csv \
    --output-dir output

Input CSV from model.py should have columns:
  - Tract_GEOID
  - Age_Group (e.g., '45to49', '65to69')
  - Race_Ethnicity (e.g., 'White_NonHispanic', 'Hispanic_Latino')
  - Income_Bracket (e.g., 'Less10k', '20to25k', '75to100k')
  - Health_Insurance_Status (e.g., 'Insured', 'Uninsured')
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from screening_calculator import ScreeningCalculator

logger = logging.getLogger(__name__)


class MedicaidPolicySimulator:
    """
    CSV-driven Medicaid coverage policy simulator.

    Infers likely Medicaid status from synthetic population data and applies
    policy rules defined in external CSV files, then (optionally) recomputes
    screening based on post-policy insurance status.

    Designed to work with the synthetic population output from model.py.
    """

    # 2025 Federal Poverty Line (48 states) – reference values
    FPL_INDIVIDUAL = 15000  # Annual
    FPL_FAMILY_3 = 31720    # Family of 3
    FPL_FAMILY_4 = 38775    # Family of 4

    def __init__(self,
                 screening_joint_distributions_csv: str,
                 colon_rates_csv: str):
        """Initialize policy simulator + reusable screening calculator."""
        logger.info("Initialized MedicaidPolicySimulator (CSV-driven)")
        self.screening_calculator = ScreeningCalculator(
            screening_joint_distributions_csv=screening_joint_distributions_csv,
            colon_rates_csv=colon_rates_csv,
        )

    # ========================================================================
    # INCOME BRACKET MAPPING (model.py → policy brackets)
    # ========================================================================

    def map_model_income_to_policy_bracket(self, income_bracket: str) -> str:
        """
        Map model.py Income_Bracket (ACS-style) to coarser policy brackets.

        model.py brackets:
          Less10k, 10to15k, 15to20k, 20to25k, 25to30k, 30to35k,
          35to40k, 40to45k, 45to50k, 50to60k, 60to75k,
          75to100k, 100to125k, 125to150k, 150to200k, 200kplus

        Policy brackets (coarse):
          Less10k, 10to25k, 25to50k, 50to75k, 75to100k, 100kplus
        """
        if income_bracket in ['Less10k']:
            return 'Less10k'
        elif income_bracket in ['10to15k', '15to20k', '20to25k']:
            return '10to25k'
        elif income_bracket in ['25to30k', '30to35k', '35to40k',
                                '40to45k', '45to50k']:
            return '25to50k'
        elif income_bracket in ['50to60k', '60to75k']:
            return '50to75k'
        elif income_bracket in ['75to100k']:
            return '75to100k'
        else:
            # 100to125k, 125to150k, 150to200k, 200kplus and any unknown
            return '100kplus'

    # ========================================================================
    # MEDICAID INFERENCE LOGIC
    # ========================================================================

    def infer_medicaid_status(self, row: pd.Series) -> Tuple[bool, str]:
        """
        Infer if individual likely has Medicaid based on demographics.

        Logic:
        - If Uninsured: not Medicaid
        - If Insured + Low Income (< 138% FPL ≈ $20.7k): likely Medicaid
        - If Insured + Higher Income: likely private/employer insurance
        """
        insurance_status = row['Health_Insurance_Status']
        income_bracket_model = row['Income_Bracket']

        if insurance_status == 'Uninsured':
            return False, "Uninsured"

        policy_bracket = self.map_model_income_to_policy_bracket(income_bracket_model)
        estimated_annual_income = self.get_income_bracket_value(policy_bracket)

        if estimated_annual_income < 25000 and insurance_status == 'Insured':
            return True, f"Low income (${estimated_annual_income}) + Insured"

        return False, f"Income (${estimated_annual_income}) above Medicaid threshold"

    def get_income_bracket_value(self, income_bracket: str) -> float:
        """Get midpoint annual income for income bracket (policy brackets)."""
        income_map = {
            'Less10k': 5000,
            '10to25k': 17500,
            '25to50k': 37500,
            '50to75k': 62500,
            '75to100k': 87500,
            '100kplus': 150000,
        }
        return income_map.get(income_bracket, 50000)

    # ========================================================================
    # NAMED CONDITION HANDLERS (for use in CSV-driven policies)
    # ========================================================================

    def _cond_income_between_100_138_fpl(self, df: pd.DataFrame) -> pd.Series:
        """
        Mask for people with estimated income between 100% and 138% FPL.
        Targets individuals earning $25.7k–$35.6k who would lose Medicaid
        under a 100% FPL threshold reduction.
        """
        policy_brackets = df['Income_Bracket'].apply(self.map_model_income_to_policy_bracket)
        incomes = policy_brackets.apply(self.get_income_bracket_value)
        fpl_100 = 25700
        fpl_138 = 35600
        return (incomes > fpl_100) & (incomes < fpl_138)

    def _cond_age_18_55_random(self, df: pd.DataFrame, rate: float) -> pd.Series:
        """
        Mask for working-age adults (18–55) with Bernoulli(rate) selection.

        Used for modeling administrative churn: a fraction of eligible
        working-age enrollees lose coverage due to paperwork/barriers.
        """
        ages = df['Age_Group'].apply(self._extract_age_from_group)
        base_mask = (ages >= 18) & (ages <= 55)

        np.random.seed(42)
        r = np.random.rand(len(df))
        return base_mask & (r < rate)

    def _cond_immigrant_proxy(self, df: pd.DataFrame) -> pd.Series:
        """
        Mask for individuals estimated as immigrants using race/ethnicity proxy.
        """
        immigrant_prob_by_race = {
            'Hispanic_Latino': 0.35,
            'Asian_NonHispanic': 0.55,
            'Asian': 0.55,
            'SomeOther_NonHispanic': 0.30,
            'Black_NonHispanic': 0.05,
            'White_NonHispanic': 0.08,
            'TwoOrMore_NonHispanic': 0.15,
            'AIAN_NonHispanic': 0.10,
            'NHOPI_NonHispanic': 0.10,
        }
        races = df['Race_Ethnicity']
        probs = races.map(lambda r: immigrant_prob_by_race.get(r, 0.10))

        np.random.seed(42)
        draws = np.random.rand(len(df))
        return draws < probs

    # ========================================================================
    # CSV-DRIVEN POLICY APPLICATION
    # ========================================================================

    def apply_policy_from_config(self, pop_df: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a Medicaid coverage policy defined in a CSV config file.

        Multiple rows are OR-ed: first matching rule wins.
        """
        result = pop_df.copy()

        if 'Medicaid_Status' not in result.columns:
            result['Medicaid_Status'] = result.apply(
                self.infer_medicaid_status, axis=1
            ).apply(lambda x: x[0])

        result['Original_Insurance'] = result['Health_Insurance_Status']

        changes = [
            {'change': row['Health_Insurance_Status'], 'notes': 'No change'}
            for _, row in result.iterrows()
        ]

        for _, rule in policy_df.iterrows():
            target_field = rule['Target_Field']
            cond = rule['Condition']
            value = rule['Value']
            op = rule.get('Operator', 'AND')
            action = rule['Action']
            coverage_change = rule['Coverage_Change']
            note = rule['Note']

            if pd.isna(action) or action == '':
                continue

            if 'Income_Between_100_138_FPL' in str(action):
                mask_action = self._cond_income_between_100_138_fpl(result)
            elif 'Age_18_55_Random_' in str(action):
                try:
                    rate = float(str(action).split('_')[-1])
                    mask_action = self._cond_age_18_55_random(result, rate)
                except (ValueError, IndexError):
                    continue
            elif 'Immigrant_Proxy' in str(action):
                mask_action = self._cond_immigrant_proxy(result)
            else:
                logger.warning(f"Unknown action in policy: {action}")
                continue

            if cond == '==' and str(value).lower() == 'true':
                mask_target = (result[target_field] == True)
            elif cond == '==' and str(value).lower() == 'false':
                mask_target = (result[target_field] == False)
            elif cond == '!=':
                mask_target = (result[target_field] != value)
            else:
                mask_target = (result[target_field].astype(str) == str(value))

            if op == 'AND':
                mask = mask_target & mask_action
            else:
                mask = mask_target & mask_action

            for idx in result[mask].index:
                if changes[idx]['change'] == result.loc[idx, 'Health_Insurance_Status']:
                    if coverage_change == 'Uninsured':
                        changes[idx]['change'] = 'Uninsured'
                    elif coverage_change in ['No_Change', 'Keep']:
                        changes[idx]['change'] = result.loc[idx, 'Health_Insurance_Status']
                    else:
                        changes[idx]['change'] = coverage_change
                    changes[idx]['notes'] = note

        result['Policy_Coverage_Change'] = [c['change'] for c in changes]
        result['Policy_Notes'] = [c['notes'] for c in changes]
        result['Coverage_Status_After_Policy'] = result['Policy_Coverage_Change']

        return result

    def apply_baseline(self, pop_df: pd.DataFrame) -> pd.DataFrame:
        """
        Baseline scenario: current Medicaid expansion (138% FPL).
        No coverage changes; keeps existing insurance and screening.
        """
        result = pop_df.copy()
        if 'Medicaid_Status' not in result.columns:
            result['Medicaid_Status'] = result.apply(
                self.infer_medicaid_status, axis=1
            ).apply(lambda x: x[0])

        result['Policy_Coverage_Change'] = 'No Change'
        result['Policy_Notes'] = 'Baseline - 138% FPL threshold maintained'
        result['Coverage_Status_After_Policy'] = result['Health_Insurance_Status']

        # Recompute screening using the SAME logic, but baseline insurance
        result = self.screening_calculator.assign_screening_to_population(
            result,
            insurance_column='Coverage_Status_After_Policy'
        )

        return result

    # ========================================================================
    # REPORTING AND COMPARISON
    # ========================================================================

    def _generate_comparison_report(self, scenarios: Dict) -> Dict:
        """
        Generate comparison report across all policy scenarios.
        """
        comparison = {}
        for scenario_name, scenario_df in scenarios.items():
            insured_count = (scenario_df['Coverage_Status_After_Policy'] == 'Insured').sum()
            uninsured_count = (scenario_df['Coverage_Status_After_Policy'] == 'Uninsured').sum()
            total = len(scenario_df)

            medicaid_estimated = 0
            if 'Medicaid_Status' in scenario_df.columns:
                medicaid_estimated = (scenario_df['Medicaid_Status'] == True).sum()

            coverage_change_to_uninsured = 0
            if 'Policy_Coverage_Change' in scenario_df.columns:
                coverage_change_to_uninsured = (scenario_df['Policy_Coverage_Change'] == 'Uninsured').sum()

            comparison[scenario_name] = {
                'total_population': total,
                'insured': insured_count,
                'uninsured': uninsured_count,
                'insured_pct': 100 * insured_count / total if total else 0,
                'uninsured_pct': 100 * uninsured_count / total if total else 0,
                'medicaid_estimated': medicaid_estimated,
                'coverage_loss': coverage_change_to_uninsured,
                'coverage_loss_pct': 100 * coverage_change_to_uninsured / total if total else 0,
            }

            print(f"\n{scenario_name}:")
            print(f"  Total population: {total}")
            print(f"  Insured: {insured_count} ({comparison[scenario_name]['insured_pct']:.1f}%)")
            print(f"  Uninsured: {uninsured_count} ({comparison[scenario_name]['uninsured_pct']:.1f}%)")
            if coverage_change_to_uninsured > 0:
                print(f"  Coverage losses: {coverage_change_to_uninsured} "
                      f"({comparison[scenario_name]['coverage_loss_pct']:.2f}%)")

        return comparison

    def _extract_age_from_group(self, age_group: str) -> int:
        """Approximate age from age group string."""
        age_map = {
            'Under5': 2,
            '5to9': 7,
            '10to14': 12,
            '15to17': 16,
            '18to19': 18,
            '20': 20,
            '21': 21,
            '22to24': 23,
            '25to29': 27,
            '30to34': 32,
            '35to39': 37,
            '40to44': 42,
            '45to49': 47,
            '50to54': 52,
            '55to59': 57,
            '60to61': 60,
            '62to64': 63,
            '65to66': 65,
            '67to69': 68,
            '70to74': 72,
            '75to79': 77,
            '80to84': 82,
            '85plus': 87,
        }
        return age_map.get(age_group, 40)


# ============================================================================
# MAIN: CSV-DRIVEN POLICY SIMULATION
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Medicaid Policy Simulator (CSV-driven, screening-aware)"
    )
    parser.add_argument(
        "--population", required=True,
        help="Synthetic population CSV from model.py (e.g., synthetic_population.csv)"
    )
    parser.add_argument(
        "--screening-joint-dist", required=True,
        help="Screening joint distributions CSV (same as model.py Stage 2)"
    )
    parser.add_argument(
        "--colon-rates", required=True,
        help="Colon screening rates CSV (same as model.py Stage 2)"
    )
    parser.add_argument(
        "--policy", action="append", required=False,
        help="Policy CSV file (can specify multiple --policy options). Omit for baseline only."
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory to save scenario outputs and summary (default: current dir)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MEDICAID POLICY SIMULATOR (CSV-Driven + Screening Recalculation)")
    print("=" * 80)

    print(f"\nLoading synthetic population from: {args.population}")
    pop_df = pd.read_csv(args.population)
    print(f"✓ Loaded {len(pop_df)} individuals")

    required_cols = ['Income_Bracket', 'Health_Insurance_Status', 'Age_Group',
                     'Race_Ethnicity', 'Tract_GEOID']
    missing = [c for c in required_cols if c not in pop_df.columns]
    if missing:
        raise ValueError(f"Input file missing required columns from model.py: {missing}")

    simulator = MedicaidPolicySimulator(
        screening_joint_distributions_csv=args.screening_joint_dist,
        colon_rates_csv=args.colon_rates,
    )

    scenarios = {}

    # Baseline scenario (always included)
    print("\n" + "-" * 80)
    print("Applying BASELINE policy (138% FPL threshold, no changes)...")
    baseline_df = simulator.apply_baseline(pop_df)
    scenarios['Baseline'] = baseline_df
    baseline_path = f"{args.output_dir}/population_medicaid_Baseline.csv"
    baseline_df.to_csv(baseline_path, index=False)
    print(f"✓ Saved: {baseline_path}")

    # Apply each policy CSV
    if args.policy:
        for policy_path in args.policy:
            print("\n" + "-" * 80)
            print(f"Loading policy from: {policy_path}")
            policy_df = pd.read_csv(policy_path)

            if len(policy_df) == 0:
                print("⚠ Policy file is empty; skipping")
                continue

            policy_name = policy_df['Policy_Name'].iloc[0] if 'Policy_Name' in policy_df.columns else policy_path
            print(f"Applying policy: {policy_name}")

            policy_result = simulator.apply_policy_from_config(pop_df, policy_df)

            # Recompute screening after policy based on Coverage_Status_After_Policy
            policy_result = simulator.screening_calculator.assign_screening_to_population(
                policy_result,
                insurance_column='Coverage_Status_After_Policy'
            )

            scenarios[policy_name] = policy_result

            out_file = f"{args.output_dir}/population_medicaid_{policy_name}.csv"
            policy_result.to_csv(out_file, index=False)
            print(f"✓ Saved: {out_file}")
    else:
        print("\nℹ No policy CSVs specified (--policy). Running baseline only.")

    # Generate and display comparison
    print("\n" + "=" * 80)
    print("POLICY COMPARISON SUMMARY")
    print("=" * 80)
    comparison = simulator._generate_comparison_report(scenarios)

    comparison_df = pd.DataFrame(comparison).T
    summary_path = f"{args.output_dir}/policy_comparison_summary.csv"
    comparison_df.to_csv(summary_path)
    print(f"\n✓ Saved comparison summary: {summary_path}\n")
    print(comparison_df.to_string())
    print("\n" + "=" * 80)
