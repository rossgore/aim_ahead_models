"""
INTEGRATED SYNTHETIC POPULATION PIPELINE - CANCER-AGNOSTIC VERSION

Combines three sequential stages into a single, unified pipeline:
1. Generate synthetic population using IPF with dedicated joint distributions
2. Assign cancer screening status INDEPENDENT with its own joint distributions
3. Calculate cancer risk assessment INDEPENDENT (CCRAT for colon, BCRAT for breast)

KEY FEATURES:
- Supports both colorectal and breast cancer screening simulations
- Cancer type specified via --cancer-type command line argument
- Automatically loads appropriate parameter files for each cancer type
- Uses refactored ScreeningCalculator (cancer-agnostic)
- Independent screening and risk assessment stages

Usage:
    # Colon cancer
    python model.py --cancer-type colon \
        --demographics data/demographics.csv \
        --ipf-joint-dist data/ipf-joint-distributions.csv \
        --screening-joint-dist data/colon-screening-joint-distributions.csv \
        --screening-rates data/colon-rates.csv \
        --risk-parameters data/ccrat-parameters.csv \
        --output output/synthetic_population_colon.csv

    # Breast cancer
    python model.py --cancer-type breast \
        --demographics data/demographics.csv \
        --ipf-joint-dist data/ipf-joint-distributions.csv \
        --screening-joint-dist data/breast-screening-joint-distributions.csv \
        --screening-rates data/breast-cancer-rates.csv \
        --risk-parameters data/bcrat-parameters.csv \
        --output output/synthetic_population_breast.csv
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from screening_calculator import ScreeningCalculator
from modality_assigner import ModalityAssigner

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class IntegratedSyntheticPopulationPipeline:
    """
    Unified pipeline for generating synthetic population with screening and risk.

    Uses ScreeningCalculator for all screening logic to avoid duplication.
    Supports both colon and breast cancer with appropriate age ranges and parameters.
    """

    # Age group mappings by cancer type
    CANCER_AGE_MAPPINGS = {
        'colon': {
            '40to44': None,  # Not eligible for colon screening
            '45to49': '45to49',
            '50to54': '50to54',
            '55to59': '55to59',
            '60to61': '60to64',
            '62to64': '60to64',
            '65to66': '65to69',
            '67to69': '65to69',
            '70to74': '70to75',
            '75to79': '70to75',
            '80to84': '70to75',
            '85plus': '70to75'
        },
        'breast': {
            '40to44': '40to44',
            '45to49': '45to49',
            '50to54': '50to54',
            '55to59': '55to59',
            '60to61': '60to64',
            '62to64': '60to64',
            '65to66': '65to69',
            '67to69': '65to69',
            '70to74': '70to74',
            '75to79': None,  # Not eligible for breast screening
            '80to84': None,
            '85plus': None
        }
    }

    def __init__(
        self,
        cancer_type: str,
        demographics_csv: str,
        ipf_joint_distributions_csv: str,
        screening_joint_distributions_csv: str,
        screening_rates_csv: str,
        risk_parameters_csv: str,
        scaling_factor: int = 100
    ):
        """
        Initialize the integrated pipeline.

        Args:
            cancer_type: 'colon' or 'breast'
            demographics_csv: ACS tract-level population data
            ipf_joint_distributions_csv: For Stage 1 synthetic population generation
            screening_joint_distributions_csv: For Stage 2 screening status assignment
            screening_rates_csv: Tract-level screening rates
            risk_parameters_csv: CCRAT or BCRAT parameters
            scaling_factor: Population scaling (default 100)
        """
        if cancer_type.lower() not in ['colon', 'breast']:
            raise ValueError(f"cancer_type must be 'colon' or 'breast', got: {cancer_type}")

        self.cancer_type = cancer_type.lower()
        self.age_mapping = self.CANCER_AGE_MAPPINGS[self.cancer_type]

        logger.info("=" * 80)
        logger.info(f"INITIALIZING INTEGRATED SYNTHETIC POPULATION PIPELINE")
        logger.info(f"Cancer Type: {self.cancer_type.upper()}")
        logger.info("=" * 80)

        try:
            logger.info(f"1. Loading demographics from {demographics_csv}")
            self.demographics_df = pd.read_csv(demographics_csv)
            logger.info(f"   Loaded {len(self.demographics_df)} census tracts")

            logger.info(f"2. Loading IPF joint distributions from {ipf_joint_distributions_csv}")
            self.ipf_joint_dist_df = pd.read_csv(ipf_joint_distributions_csv)
            logger.info(f"   Loaded IPF joint distributions for {len(self.ipf_joint_dist_df)} tracts")

            logger.info(f"3. Initializing Screening Calculator ({self.cancer_type} cancer)...")
            self.screening_calculator = ScreeningCalculator(
                cancer_type=self.cancer_type,
                screening_joint_distributions_csv=screening_joint_distributions_csv,
                screening_rates_csv=screening_rates_csv
            )

            logger.info(f"4. Loading {self.cancer_type.upper()} risk parameters from {risk_parameters_csv}")
            self.risk_parameters_df = pd.read_csv(risk_parameters_csv)
            logger.info(f"   Loaded {len(self.risk_parameters_df)} risk parameters")

            self.risk_parameters = self._parse_risk_parameters(self.risk_parameters_df)
            logger.info(f"   Parsed parameter categories: {', '.join(self.risk_parameters.keys())}")

            logger.info("5. Preparing Stage 1 dataset (demographics + IPF joint distributions)...")
            ipf_subset = self.ipf_joint_dist_df.copy()
            self.stage1_df = self.demographics_df.merge(ipf_subset, on='GEOID', how='left')
            logger.info(f"   Stage 1 dataset has shape {self.stage1_df.shape}")

            self.scaling_factor = scaling_factor

            # Define demographic categories
            self.age_groups = [
                'Under5', '5to9', '10to14', '15to17', '18to19', '20', '21', '22to24',
                '25to29', '30to34', '35to39', '40to44', '45to49', '50to54', '55to59',
                '60to61', '62to64', '65to66', '67to69', '70to74', '75to79', '80to84', '85plus'
            ]

            self.races = [
                'White_NonHispanic', 'Black_NonHispanic', 'Hispanic_Latino',
                'Asian_NonHispanic', 'AIAN_NonHispanic', 'NHOPI_NonHispanic',
                'SomeOther_NonHispanic', 'TwoOrMore_NonHispanic'
            ]

            self.income_brackets = [
                'Less10k', '10to15k', '15to20k', '20to25k', '25to30k', '30to35k',
                '35to40k', '40to45k', '45to50k', '50to60k', '60to75k', '75to100k',
                '100to125k', '125to150k', '150to200k', '200kplus'
            ]

            self.education_levels = [
                'Lessthan9thGrade', '9thto12thGradeNoDiploma', 'HighSchoolGraduate',
                'SomeCollegeNoDegree', 'AssociatesDegree', 'BachelorsDegree',
                'MastersDegree', 'ProfessionalBeyondMasters'
            ]

            logger.info("✓ Pipeline initialized successfully")

        except Exception as e:
            logger.error(f"ERROR during initialization: {e}")
            raise

    def _parse_risk_parameters(self, df: pd.DataFrame) -> dict:
        """Parse risk parameters CSV into dictionaries for easy lookup."""
        params = {}

        for category in df['Parameter_Category'].unique():
            category_data = df[df['Parameter_Category'] == category]

            if category == 'age_baseline_risk':
                params['age_baseline_risk'] = {}
                for _, row in category_data.iterrows():
                    age = row['Parameter_Name'].replace('risk_', '')
                    params['age_baseline_risk'][age] = row['Parameter_Value']

            elif category == 'gender_multiplier':
                params['gender_multiplier'] = {}
                for _, row in category_data.iterrows():
                    gender = row['Parameter_Name'].title()
                    params['gender_multiplier'][gender] = row['Parameter_Value']

            elif category == 'race_multiplier':
                params['race_multiplier'] = {}
                for _, row in category_data.iterrows():
                    race = row['Parameter_Name'].title()
                    params['race_multiplier'][race] = row['Parameter_Value']

            elif category == 'income_multiplier':
                params['income_multiplier'] = {}
                for _, row in category_data.iterrows():
                    income = row['Parameter_Name']
                    params['income_multiplier'][income] = row['Parameter_Value']

            elif category == 'education_multiplier':
                params['education_multiplier'] = {}
                for _, row in category_data.iterrows():
                    education = row['Parameter_Name']
                    params['education_multiplier'][education] = row['Parameter_Value']

            elif category == 'screening_effectiveness':
                params['screening_effectiveness'] = category_data.iloc[0]['Parameter_Value']

        return params

    def _get_marginal_age_gender(self, row) -> Dict:
        """Extract age-gender marginal distribution from census data."""
        age_gender_probs = {}
        for age_group in self.age_groups:
            male_col = f"Male_{age_group}"
            female_col = f"Female_{age_group}"
            male_count = row.get(male_col, 0) or 0
            female_count = row.get(female_col, 0) or 0
            age_gender_probs[(age_group, 'M')] = float(male_count)
            age_gender_probs[(age_group, 'F')] = float(female_count)

        total = sum(age_gender_probs.values())
        if total > 0:
            age_gender_probs = {k: v/total for k, v in age_gender_probs.items()}
        return age_gender_probs

    def _get_marginal_race(self, row) -> Dict:
        """Extract race marginal distribution from census data."""
        race_probs = {}
        for race in self.races:
            count = row.get(race, 0) or 0
            race_probs[race] = float(count)

        total = sum(race_probs.values())
        if total > 0:
            race_probs = {k: v/total for k, v in race_probs.items()}
        return race_probs

    def _get_joint_income_given_race_ipf(self, row, race: str) -> Dict:
        """Get P(Income | Race) from IPF joint distributions."""
        income_probs = {}
        for bracket in self.income_brackets:
            col_name = f"{race}_P(Income={bracket})"
            prob = row.get(col_name, 1.0 / len(self.income_brackets))
            income_probs[bracket] = max(0, float(prob))

        total = sum(income_probs.values())
        if total > 0:
            income_probs = {k: v/total for k, v in income_probs.items()}
        return income_probs

    def _get_joint_education_given_race_ipf(self, row, race: str) -> Dict:
        """Get P(Education | Race) from IPF joint distributions."""
        edu_probs = {}
        for edu_level in self.education_levels:
            col_name = f"{race}_P(Education={edu_level})"
            prob = row.get(col_name, 1.0 / len(self.education_levels))
            edu_probs[edu_level] = max(0, float(prob))

        total = sum(edu_probs.values())
        if total > 0:
            edu_probs = {k: v/total for k, v in edu_probs.items()}
        return edu_probs

    def _get_joint_insurance_given_race_age_ipf(self, row, race: str, age_group: str) -> Dict:
        """Get P(Insurance | Race, Age) from IPF joint distributions."""
        age_for_insurance = 'Over65' if age_group in [
            '65to66', '67to69', '70to74', '75to79', '80to84', '85plus'
        ] else 'Under65'

        col_name = f"{race}_P(Insurance={age_for_insurance}=Insured)"
        insured_prob = row.get(col_name, None)

        if insured_prob is None or pd.isna(insured_prob):
            fallback_col = f"P(Insurance={age_for_insurance})"
            insured_prob = row.get(fallback_col, 0.89 if age_for_insurance == 'Under65' else 0.975)

        if pd.isna(insured_prob) or insured_prob <= 0 or insured_prob >= 1:
            insured_prob = 0.89 if age_for_insurance == 'Under65' else 0.975

        return {'Insured': insured_prob, 'Uninsured': 1.0 - insured_prob}

    def _ipf_fit(self, sample_size: int, row, max_iterations: int = 20, tolerance: float = 0.001):
        """Iterative Proportional Fitting to generate synthetic individuals."""
        age_gender_target = self._get_marginal_age_gender(row)
        race_target = self._get_marginal_race(row)

        for iteration in range(max_iterations):
            individuals = []

            for person_id in range(sample_size):
                # Sample age-gender
                age_gender_options = [(k, v) for k, v in age_gender_target.items() if v > 0]
                if age_gender_options:
                    (age_group, gender), _ = max(age_gender_options, 
                                                 key=lambda x: x[1] + np.random.exponential(0.1))
                else:
                    age_group, gender = ('25to29', 'M')

                # Sample race
                race_options = [(k, v) for k, v in race_target.items() if v > 0]
                if race_options:
                    race, _ = max(race_options, key=lambda x: x[1] + 0)
                else:
                    race = 'White_NonHispanic'

                # Sample income given race
                income_dist = self._get_joint_income_given_race_ipf(row, race)
                income_options = [(k, v) for k, v in income_dist.items() if v > 0]
                income = max(income_options, key=lambda x: x[1] + 0)[0] if income_options else 'Less10k'

                # Sample education given race
                is_25plus = age_group not in [
                    'Under5', '5to9', '10to14', '15to17', '18to19', '20', '21', '22to24'
                ]
                if is_25plus:
                    edu_dist = self._get_joint_education_given_race_ipf(row, race)
                    edu_options = [(k, v) for k, v in edu_dist.items() if v > 0]
                    education = max(edu_options, key=lambda x: x[1] + 0)[0] if edu_options else 'HighSchoolGraduate'
                else:
                    education = 'Under25'

                # Sample insurance given race and age
                insurance_dist = self._get_joint_insurance_given_race_age_ipf(row, race, age_group)
                insurance = np.random.choice(
                    ['Insured', 'Uninsured'],
                    p=[insurance_dist['Insured'], insurance_dist['Uninsured']]
                )

                individuals.append({
                    'age_group': age_group,
                    'gender': gender,
                    'race': race,
                    'income': income,
                    'education': education,
                    'insurance': insurance
                })

            synth_df = pd.DataFrame(individuals)

            # Check convergence
            age_gender_synth = {}
            for age_group in self.age_groups:
                for gender in ['M', 'F']:
                    count = len(synth_df[(synth_df['age_group'] == age_group) & 
                                        (synth_df['gender'] == gender)])
                    age_gender_synth[(age_group, gender)] = count / sample_size if sample_size > 0 else 0

            race_synth = {}
            for race in self.races:
                count = len(synth_df[synth_df['race'] == race])
                race_synth[race] = count / sample_size if sample_size > 0 else 0

            age_gender_rmse = np.sqrt(np.mean([
                (age_gender_target.get(k, 0) - age_gender_synth.get(k, 0))**2 
                for k in age_gender_target.keys()
            ]))
            race_rmse = np.sqrt(np.mean([
                (race_target.get(k, 0) - race_synth.get(k, 0))**2 
                for k in race_target.keys()
            ]))

            if age_gender_rmse < tolerance and race_rmse < tolerance:
                break

        return pd.DataFrame(individuals)

    def generate_synthetic_population(self, sample_size: int = 100) -> pd.DataFrame:
        """Generate synthetic individuals for all tracts using IPF."""
        logger.info("=" * 80)
        logger.info("STAGE 1: GENERATING SYNTHETIC POPULATION (IPF with IPF Joint Distributions)")
        logger.info("=" * 80)

        synthetic_individuals = []

        for idx, row in self.stage1_df.iterrows():
            tract_id = row['GEOID']
            tract_name = row.get('NAME_y', row.get('Tract_Name', 'Unknown'))
            total_pop = row.get('Total_Population', 0)

            if pd.isna(total_pop) or total_pop == 0:
                logger.warning(f"  Skipping {tract_name} - no population data")
                continue

            race_sum = row[[r for r in self.races if r in row.index]].sum() if all(r in row.index for r in self.races) else 0
            if race_sum == 0:
                logger.warning(f"  {tract_name} - no demographic data available")
                continue

            tract_sample_size = max(5, int(total_pop / self.scaling_factor))

            if (idx + 1) % 50 == 0:
                logger.info(f"  Processing tract {idx + 1}/{len(self.stage1_df)}: {tract_name}")

            tract_individuals = self._ipf_fit(tract_sample_size, row)
            tract_individuals['Tract_GEOID'] = tract_id
            tract_individuals['Tract_Name'] = tract_name
            tract_individuals['Individual_ID'] = [f"{tract_id}_{i:04d}" for i in range(len(tract_individuals))]
            tract_individuals['Tract_Total_Pop'] = total_pop
            tract_individuals['Median_Household_Income'] = row.get('Median_Household_Income', np.nan)

            # Rename columns
            tract_individuals = tract_individuals.rename(columns={
                'age_group': 'Age_Group',
                'gender': 'Gender',
                'race': 'Race_Ethnicity',
                'income': 'Income_Bracket',
                'education': 'Education_Level',
                'insurance': 'Health_Insurance_Status'
            })

            tract_individuals = tract_individuals[[
                'Tract_GEOID', 'Tract_Name', 'Individual_ID', 'Age_Group', 'Gender',
                'Race_Ethnicity', 'Income_Bracket', 'Education_Level',
                'Health_Insurance_Status', 'Tract_Total_Pop', 'Median_Household_Income'
            ]]

            synthetic_individuals.append(tract_individuals)

        result = pd.concat(synthetic_individuals, ignore_index=True) if synthetic_individuals else pd.DataFrame()
        logger.info(f"✓ Generated {len(result)} synthetic individuals")
        return result

    def assign_screening_status(
        self, 
        population_df: pd.DataFrame,
        insurance_column: str = 'Health_Insurance_Status'
    ) -> pd.DataFrame:
        """Assign screening status using the reusable ScreeningCalculator."""
        return self.screening_calculator.assign_screening_to_population(
            population_df, 
            insurance_column=insurance_column
        )

    def _map_age_group(self, age_group: str) -> Optional[str]:
        """Map Age_Group to cancer-specific risk assessment age category."""
        return self.age_mapping.get(age_group, None)

    def _map_gender(self, gender: str) -> str:
        """Map Gender to risk assessment format."""
        gender_mapping = {'M': 'Male', 'F': 'Female'}
        return gender_mapping.get(gender, 'Male')

    def _map_race(self, race_ethnicity: str) -> str:
        """Map Race_Ethnicity to risk assessment format."""
        race_mapping = {
            'White_NonHispanic': 'White',
            'Black_NonHispanic': 'Black',
            'Hispanic_Latino': 'Hispanic',
            'Asian_NonHispanic': 'Asian',
            'AIAN_NonHispanic': 'Aian',
            'NHOPI_NonHispanic': 'Nhopi',
            'SomeOther_NonHispanic': 'Someother',
            'TwoOrMore_NonHispanic': 'Twoormore'
        }
        return race_mapping.get(race_ethnicity, 'Other')

    def _map_income(self, income_bracket: str) -> str:
        """Map Income_Bracket to risk assessment format."""
        income_mapping = {
            'Less10k': 'under20k',
            '10to15k': 'under20k',
            '15to20k': 'under20k',
            '20to25k': '20kto35k',
            '25to30k': '20kto35k',
            '30to35k': '20kto35k',
            '35to40k': '35kto50k',
            '40to45k': '35kto50k',
            '45to50k': '35kto50k',
            '50to60k': '50kto75k',
            '60to75k': '50kto75k',
            '75to100k': 'over75k',
            '100to125k': 'over75k',
            '125to150k': 'over75k',
            '150to200k': 'over75k',
            '200kplus': 'over75k'
        }
        return income_mapping.get(income_bracket, '50kto75k')

    def _map_education(self, education_level: str) -> str:
        """Map Education_Level to risk assessment format."""
        education_mapping = {
            'Under25': 'somecollege',
            'Lessthan9thGrade': 'lessthanhs',
            '9thto12thGradeNoDiploma': 'lessthanhs',
            'HighSchoolGraduate': 'hsgraduate',
            'SomeCollegeNoDegree': 'somecollege',
            'AssociatesDegree': 'somecollege',
            'BachelorsDegree': 'bachelorplus',
            'MastersDegree': 'bachelorplus',
            'ProfessionalBeyondMasters': 'bachelorplus'
        }
        return education_mapping.get(education_level, 'hsgraduate')

    def _calculate_risk(
        self, 
        age_group: str, 
        gender: str, 
        race_ethnicity: str, 
        income_bracket: str, 
        education_level: str
    ) -> Optional[Dict]:
        """Calculate cancer risk for individual INDEPENDENT of screening status."""
        cancer_age = self._map_age_group(age_group)
        if cancer_age is None:
            return None

        cancer_gender = self._map_gender(gender)
        cancer_race = self._map_race(race_ethnicity)
        cancer_income = self._map_income(income_bracket)
        cancer_education = self._map_education(education_level)

        # Get baseline risk
        baseline_risk = self.risk_parameters['age_baseline_risk'].get(cancer_age, 0.5)

        # Get multipliers
        gender_mult = self.risk_parameters['gender_multiplier'].get(cancer_gender, 1.0)
        race_mult = self.risk_parameters['race_multiplier'].get(cancer_race, 1.0)
        income_mult = self.risk_parameters['income_multiplier'].get(cancer_income, 1.0)
        education_mult = self.risk_parameters['education_multiplier'].get(cancer_education, 1.0)

        # Calculate risks
        combined_multiplier = gender_mult * race_mult * income_mult * education_mult
        unscreened_risk = baseline_risk * combined_multiplier

        screening_effectiveness = self.risk_parameters['screening_effectiveness']
        screened_risk = unscreened_risk * (1 - screening_effectiveness)

        return {
            'Unscreened_Risk': unscreened_risk,
            'Screened_Risk': screened_risk,
            'Screening_Benefit': unscreened_risk - screened_risk,
            'Baseline_Risk': baseline_risk,
            'Gender_Multiplier': gender_mult,
            'Race_Multiplier': race_mult,
            'Income_Multiplier': income_mult,
            'Education_Multiplier': education_mult,
            'Risk_Category': self._categorize_risk(unscreened_risk)
        }

    def _categorize_risk(self, risk: float) -> str:
        """Categorize risk as Low, Medium, or High."""
        if risk < 0.4:
            return 'Low'
        elif risk < 0.8:
            return 'Medium'
        else:
            return 'High'

    def assess_risk(self, population_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk for entire population INDEPENDENT."""
        logger.info("=" * 80)
        logger.info(f"STAGE 3: CALCULATING RISK ASSESSMENT (INDEPENDENT - {self.cancer_type.upper()})")
        logger.info("=" * 80)

        risks = []

        for idx, row in population_df.iterrows():
            risk = self._calculate_risk(
                row['Age_Group'],
                row['Gender'],
                row['Race_Ethnicity'],
                row['Income_Bracket'],
                row['Education_Level']
            )

            if risk is not None:
                risks.append(risk)
            else:
                risks.append({
                    'Unscreened_Risk': None,
                    'Screened_Risk': None,
                    'Screening_Benefit': None,
                    'Baseline_Risk': None,
                    'Gender_Multiplier': None,
                    'Race_Multiplier': None,
                    'Income_Multiplier': None,
                    'Education_Multiplier': None,
                    'Risk_Category': 'Outside Screening Age'
                })

        risk_df = pd.DataFrame(risks)
        results = pd.concat([population_df, risk_df], axis=1)

        logger.info(f"✓ Calculated risk independently")
        return results

    def run(self, output_file: str, sample_size: int = 100) -> pd.DataFrame:
        """Execute the complete integrated pipeline with INDEPENDENT stages."""
        logger.info("=" * 80)
        logger.info("STARTING INTEGRATED PIPELINE")
        logger.info("=" * 80)

        # Stage 1: Generate population
        synth_pop = self.generate_synthetic_population(sample_size=sample_size)
        logger.info(f"✓ Stage 1 complete: {len(synth_pop)} individuals")

        # Stage 2: Assign screening
        logger.info("=" * 80)
        logger.info(f"STAGE 2: ASSIGNING SCREENING STATUS (INDEPENDENT - Using ScreeningCalculator)")
        logger.info("=" * 80)

        with_screening = self.assign_screening_status(synth_pop.copy())

        screening_col = f'{self.cancer_type.capitalize()}_Cancer_Screening_Status'
        screened = (with_screening[screening_col] == 'Screened').sum()
        logger.info(f"✓ Assigned screening status independently")
        logger.info(f" - Screened: {screened} ({screened/len(with_screening)*100:.1f}%)")
        logger.info(f" - Not screened: {len(with_screening) - screened} ({(len(with_screening) - screened)/len(with_screening)*100:.1f}%)\n")

        logger.info("Executing Stage 2b: Assigning Screening Modalities...")
    
        # Use the EXACT filename of your uploaded CSV
        modality_assigner = ModalityAssigner(
            cancer_type='colon',
            screening_modalities_csv='data/Modality-Sensitivity-Availability-Uptake-Intervalyrs-Cost.csv' 
        )
        with_screening = modality_assigner.assign_modality_to_population(with_screening)

        logger.info("=" * 80)
        logger.info("STAGE 3: CALCULATING RISK ASSESSMENT (INDEPENDENT)")
        logger.info("=" * 80)

        # Stage 3: Calculate risk
        final_results = self.assess_risk(with_screening)
        logger.info(f"✓ Stage 3 complete")

        # Save results
        logger.info("=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80)
        final_results.to_csv(output_file, index=False)
        logger.info(f"✓ Saved complete synthetic population to {output_file}")

        self._print_summary(final_results)

        return final_results

    def _print_summary(self, results: pd.DataFrame):
        """Print comprehensive summary statistics."""
        logger.info("=" * 80)
        logger.info("FINAL SUMMARY STATISTICS")
        logger.info("=" * 80)

        logger.info(f"Overview:")
        logger.info(f"  Total individuals: {len(results)}")
        logger.info(f"  Unique tracts: {results['Tract_GEOID'].nunique()}")
        logger.info(f"  Cancer type: {self.cancer_type.upper()}")

        logger.info(f"\nScreening Status (INDEPENDENT calculation with ScreeningCalculator):")
        screening_col = f'{self.cancer_type.capitalize()}_Cancer_Screening_Status'
        screened = (results[screening_col] == 'Screened').sum()
        not_screened = (results[screening_col] == 'Not_Screened').sum()
        logger.info(f"  Screened: {screened} ({screened/len(results)*100:.1f}%)")
        logger.info(f"  Not screened: {not_screened} ({not_screened/len(results)*100:.1f}%)")

        eligible_df = results[results['Age_Eligibility'].str.contains('Eligible', na=False)]
        if len(eligible_df) > 0:
            logger.info(f"\n  Among eligible ages:")
            screened_eligible = (eligible_df[screening_col] == 'Screened').sum()
            logger.info(f"    Screened: {screened_eligible} "
                       f"({screened_eligible/len(eligible_df)*100:.1f}%)")

        screening_df = results[results['Unscreened_Risk'].notna()]
        if len(screening_df) > 0:
            logger.info(f"\nRisk Distribution (INDEPENDENT calculation):")
            logger.info(f"  Mean unscreened risk: {screening_df['Unscreened_Risk'].mean():.4f}%")
            logger.info(f"  Median unscreened risk: {screening_df['Unscreened_Risk'].median():.4f}%")
            logger.info(f"  Std dev: {screening_df['Unscreened_Risk'].std():.4f}%")

            logger.info(f"\n  Risk Categories:")
            categories = screening_df['Risk_Category'].value_counts()
            for cat, count in categories.items():
                pct = count / len(screening_df) * 100
                logger.info(f"    {cat}: {count} ({pct:.1f}%)")

        logger.info("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run Integrated Synthetic Population Pipeline (Cancer-Agnostic)'
    )
    parser.add_argument('--cancer-type', required=True, choices=['colon', 'breast'],
                       help='Type of cancer (colon or breast)')
    parser.add_argument('--demographics', required=True,
                       help='Demographics CSV file')
    parser.add_argument('--ipf-joint-dist', required=True,
                       help='IPF joint distributions CSV')
    parser.add_argument('--screening-joint-dist', required=True,
                       help='Screening joint distributions CSV')
    parser.add_argument('--screening-rates', required=True,
                       help='Cancer screening rates CSV')
    parser.add_argument('--risk-parameters', required=True,
                       help='Risk parameters CSV file (CCRAT or BCRAT)')
    parser.add_argument('--output', required=True,
                       help='Output CSV file')
    parser.add_argument('--scaling-factor', type=int, default=100,
                       help='Scaling factor (default: 100)')

    args = parser.parse_args()

    pipeline = IntegratedSyntheticPopulationPipeline(
        cancer_type=args.cancer_type,
        demographics_csv=args.demographics,
        ipf_joint_distributions_csv=args.ipf_joint_dist,
        screening_joint_distributions_csv=args.screening_joint_dist,
        screening_rates_csv=args.screening_rates,
        risk_parameters_csv=args.risk_parameters,
        scaling_factor=args.scaling_factor
    )

    results = pipeline.run(output_file=args.output)
