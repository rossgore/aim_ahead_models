"""
INTEGRATED SYNTHETIC POPULATION PIPELINE (REFACTORED - DUAL JOINT DISTRIBUTIONS)
================================================================================

Combines three sequential scripts into a single, unified pipeline:
1. Generate synthetic population using IPF (with dedicated joint distributions)
2. Assign colon cancer screening status (INDEPENDENT with its own joint distributions)
3. Calculate CCRAT risk assessment (INDEPENDENT)

KEY CHANGE: Now accepts TWO separate joint_distributions CSVs:
- ipf_joint_distributions_csv: For Stage 1 (synthetic population generation)
- screening_joint_distributions_csv: For Stage 2 (screening status assignment)

Both Stages 2 and 3 operate independently from demographics.

Usage:
    pipeline = IntegratedSyntheticPopulationPipeline(
        demographics_csv="hampton-roads-with-acs-demographics.csv",
        ipf_joint_distributions_csv="ipf-joint-distributions.csv",
        screening_joint_distributions_csv="screening-joint-distributions.csv",
        colon_rates_csv="hampton_roads_colon_screen.csv",
        ccrat_parameters_csv="ccrat-parameters.csv",
        scaling_factor=100
    )
    
    results_df = pipeline.run(
        output_file="synthetic_population_complete.csv",
        sample_size=100
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class IntegratedSyntheticPopulationPipeline:
    """
    Unified pipeline ...
    """

    def __init__(self,
                 demographics_csv: str,
                 ipf_joint_distributions_csv: str,
                 screening_joint_distributions_csv: str,
                 colon_rates_csv: str,
                 ccrat_parameters_csv: str,
                 scaling_factor: int = 100):

        logger.info("=" * 80)
        logger.info("INITIALIZING INTEGRATED SYNTHETIC POPULATION PIPELINE")
        logger.info("(Dual Joint Distributions Mode)")
        logger.info("=" * 80)

        try:
            logger.info(f"\n1. Loading demographics from: {demographics_csv}")
            self.demographics_df = pd.read_csv(demographics_csv)
            logger.info(f" ✓ Loaded {len(self.demographics_df)} census tracts")
            logger.info(f"   Demographics columns: {list(self.demographics_df.columns)[:10]} ...")

            logger.info(f"\n2. Loading IPF joint distributions from: {ipf_joint_distributions_csv}")
            self.ipf_joint_dist_df = pd.read_csv(ipf_joint_distributions_csv)
            logger.info(f" ✓ Loaded IPF joint distributions for {len(self.ipf_joint_dist_df)} tracts")
            logger.info(f"   IPF joint columns: {list(self.ipf_joint_dist_df.columns)[:10]} ...")

            logger.info(f"\n3. Loading Screening joint distributions from: {screening_joint_distributions_csv}")
            self.screening_joint_dist_df = pd.read_csv(screening_joint_distributions_csv)
            logger.info(f" ✓ Loaded Screening joint distributions for {len(self.screening_joint_dist_df)} tracts")

            logger.info(f"\n4. Loading colon screening rates from: {colon_rates_csv}")
            self.colon_rates_df = pd.read_csv(colon_rates_csv)
            self.colon_rates_dict = dict(
                zip(self.colon_rates_df['GEOID'], self.colon_rates_df['COLON_SCREEN_RATE'])
            )
            logger.info(f" ✓ Loaded screening rates for {len(self.colon_rates_dict)} tracts")

            logger.info(f"\n5. Loading CCRAT parameters from: {ccrat_parameters_csv}")
            self.ccrat_parameters_df = pd.read_csv(ccrat_parameters_csv)
            logger.info(f" ✓ Loaded {len(self.ccrat_parameters_df)} CCRAT parameters")

            # Parse parameters into dictionaries for easy lookup
            self.ccrat_parameters = self._parse_ccrat_parameters(self.ccrat_parameters_df)
            logger.info(f" ✓ Parsed parameter categories: {', '.join(self.ccrat_parameters.keys())}")

            logger.info("\n6. Preparing Stage 1 dataset (demographics + IPF joint distributions)...")

            # IMPORTANT: keep GEOID to merge on; drop only duplicate non-key columns
            ipf_subset = self.ipf_joint_dist_df.copy()
            # If your IPF file has Tract_Name, keep GEOID and all others:
            # or explicitly: ipf_subset = self.ipf_joint_dist_df.drop(columns=['Tract_Name'], errors='ignore')

            self.stage1_df = self.demographics_df.merge(
                ipf_subset,
                on='GEOID',
                how='left'
            )

            logger.info(f" ✓ Stage 1 dataset has shape {self.stage1_df.shape}")

            logger.info("\n7. Preparing Stage 2 dataset (screening joint distributions)...")
            self.screening_joint_dict = dict(
                zip(self.screening_joint_dist_df['GEOID'],
                    self.screening_joint_dist_df.to_dict('records'))
            )
            logger.info(" ✓ Stage 2 screening joint distributions indexed by GEOID")

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
                'Less10k', '10to15k', '15to20k', '20to25k', '25to30k',
                '30to35k', '35to40k', '40to45k', '45to50k', '50to60k',
                '60to75k', '75to100k', '100to125k', '125to150k', '150to200k', '200kplus'
            ]

            self.education_levels = [
                'Less_than_9th_Grade', '9th_to_12th_Grade_No_Diploma',
                'High_School_Graduate', 'Some_College_No_Degree',
                'Associates_Degree', 'Bachelors_Degree',
                'Masters_Degree', 'Professional_Beyond_Masters'
            ]

            logger.info("\n✓ Pipeline initialized successfully\n")

        except Exception as e:
            logger.error(f"ERROR during initialization: {e}")
            raise


    # ============================================================================
    # STAGE 1: SYNTHETIC POPULATION GENERATION (IPF)
    # ============================================================================

    def get_marginal_age_gender(self, row) -> Dict:
        """Extract age×gender marginal distribution from census data."""
        age_gender_probs = {}
        for age_group in self.age_groups:
            male_col = f'Male_{age_group}'
            female_col = f'Female_{age_group}'
            male_count = row.get(male_col, 0) or 0
            female_count = row.get(female_col, 0) or 0
            age_gender_probs[(age_group, 'M')] = float(male_count)
            age_gender_probs[(age_group, 'F')] = float(female_count)
        
        total = sum(age_gender_probs.values())
        if total > 0:
            age_gender_probs = {k: v/total for k, v in age_gender_probs.items()}
        return age_gender_probs

    def get_marginal_race(self, row) -> Dict:
        """Extract race marginal distribution from census data."""
        race_probs = {}
        for race in self.races:
            count = row.get(race, 0) or 0
            race_probs[race] = float(count)
        
        total = sum(race_probs.values())
        if total > 0:
            race_probs = {k: v/total for k, v in race_probs.items()}
        return race_probs

    def get_joint_income_given_race_ipf(self, row, race: str) -> Dict:
        """Get P(Income | Race) from IPF joint distributions."""
        income_probs = {}
        for bracket in self.income_brackets:
            col_name = f'{race}_P_Income_{bracket}'
            prob = row.get(col_name, 1.0 / len(self.income_brackets))
            income_probs[bracket] = max(0, float(prob))
        
        total = sum(income_probs.values())
        if total > 0:
            income_probs = {k: v/total for k, v in income_probs.items()}
        return income_probs

    def get_joint_education_given_race_ipf(self, row, race: str) -> Dict:
        """Get P(Education | Race) from IPF joint distributions."""
        edu_probs = {}
        for edu_level in self.education_levels:
            col_name = f'{race}_P_Education_{edu_level}'
            prob = row.get(col_name, 1.0 / len(self.education_levels))
            edu_probs[edu_level] = max(0, float(prob))
        
        total = sum(edu_probs.values())
        if total > 0:
            edu_probs = {k: v/total for k, v in edu_probs.items()}
        return edu_probs

    def get_joint_insurance_given_race_age_ipf(self, row, race: str, age_group: str) -> Dict:
        """Get P(Insurance | Race, Age) from IPF joint distributions."""
        age_for_insurance = 'Over65' if age_group in ['65to66', '67to69', '70to74', 
                                                        '75to79', '80to84', '85plus'] else 'Under65'
        col_name = f'{race}_P_Insurance_{age_for_insurance}_Insured'
        insured_prob = row.get(col_name, None)
        
        if insured_prob is None or pd.isna(insured_prob):
            fallback_col = f'P_Insurance_{age_for_insurance}'
            insured_prob = row.get(fallback_col, 0.89 if age_for_insurance == 'Under65' else 0.975)
        
        if pd.isna(insured_prob) or insured_prob < 0 or insured_prob > 1:
            insured_prob = 0.89 if age_for_insurance == 'Under65' else 0.975
        
        return {'Insured': insured_prob, 'Uninsured': 1.0 - insured_prob}

    def ipf_fit(self, sample_size: int, row, max_iterations: int = 20, 
                tolerance: float = 0.001) -> pd.DataFrame:
        """Iterative Proportional Fitting to generate synthetic individuals using IPF joint distributions."""
        age_gender_target = self.get_marginal_age_gender(row)
        race_target = self.get_marginal_race(row)
        
        for iteration in range(max_iterations):
            individuals = []
            
            for person_id in range(sample_size):
                age_gender_options = [(k, v) for k, v in age_gender_target.items() if v > 0]
                if age_gender_options:
                    (age_group, gender), _ = max(age_gender_options,
                                                 key=lambda x: x[1] + np.random.exponential(0.1))
                else:
                    age_group, gender = '25to29', 'M'
                
                race_options = [(k, v) for k, v in race_target.items() if v > 0]
                if race_options:
                    race, _ = max(race_options,
                                 key=lambda x: x[1] + np.random.exponential(0.1))
                else:
                    race = 'White_NonHispanic'
                
                income_dist = self.get_joint_income_given_race_ipf(row, race)
                income_options = [(k, v) for k, v in income_dist.items() if v > 0]
                income = max(income_options, key=lambda x: x[1])[0] if income_options else 'Less10k'
                
                is_25plus = age_group not in ['Under5', '5to9', '10to14', '15to17', 
                                              '18to19', '20', '21', '22to24']
                if is_25plus:
                    edu_dist = self.get_joint_education_given_race_ipf(row, race)
                    edu_options = [(k, v) for k, v in edu_dist.items() if v > 0]
                    education = max(edu_options, key=lambda x: x[1])[0] if edu_options else 'High_School_Graduate'
                else:
                    education = 'Under25'
                
                insurance_dist = self.get_joint_insurance_given_race_age_ipf(row, race, age_group)
                insurance = np.random.choice(['Insured', 'Uninsured'],
                                           p=[insurance_dist['Insured'], insurance_dist['Uninsured']])
                
                individuals.append({
                    'age_group': age_group,
                    'gender': gender,
                    'race': race,
                    'income': income,
                    'education': education,
                    'insurance': insurance
                })
            
            synth_df = pd.DataFrame(individuals)
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
            
            age_gender_rmse = np.sqrt(np.mean([(age_gender_target.get(k, 0) - 
                                               age_gender_synth.get(k, 0))**2
                                              for k in age_gender_target.keys()]))
            race_rmse = np.sqrt(np.mean([(race_target.get(k, 0) - race_synth.get(k, 0))**2
                                        for k in race_target.keys()]))
            
            if age_gender_rmse < tolerance and race_rmse < tolerance:
                break
        
        return pd.DataFrame(individuals)

    def generate_synthetic_population(self, sample_size: int = 100) -> pd.DataFrame:
        """Generate synthetic individuals for all tracts using IPF with IPF joint distributions."""
        synthetic_individuals = []
        
        for idx, row in self.stage1_df.iterrows():
            tract_id = row['GEOID']
            tract_name = row.get('NAME_y', row.get('Tract_Name', 'Unknown'))
            total_pop = row.get('Total_Population', 0)
            
            if pd.isna(total_pop) or total_pop <= 0:
                logger.warning(f"  Skipping {tract_name} - no population data")
                continue
            
            race_sum = row[self.races].sum() if all(r in row.index for r in self.races) else 0
            if race_sum <= 0:
                logger.warning(f"  {tract_name} - no demographic data available")
                continue
            
            tract_sample_size = max(5, int(total_pop / self.scaling_factor))
            
            if (idx + 1) % 50 == 0:
                logger.info(f"  Processing tract {idx + 1}/{len(self.stage1_df)}: {tract_name}")
            
            tract_individuals = self.ipf_fit(tract_sample_size, row)
            
            tract_individuals['Tract_GEOID'] = tract_id
            tract_individuals['Tract_Name'] = tract_name
            tract_individuals['Individual_ID'] = [f"{tract_id}_{i:04d}" for i in range(len(tract_individuals))]
            tract_individuals['Tract_Total_Pop'] = total_pop
            tract_individuals['Median_Household_Income'] = row.get('Median_Household_Income', np.nan)
            
            tract_individuals = tract_individuals.rename(columns={
                'age_group': 'Age_Group',
                'gender': 'Gender',
                'race': 'Race_Ethnicity',
                'income': 'Income_Bracket',
                'education': 'Education_Level',
                'insurance': 'Health_Insurance_Status'
            })
            
            tract_individuals = tract_individuals[['Tract_GEOID', 'Tract_Name', 'Individual_ID',
                                                  'Age_Group', 'Gender', 'Race_Ethnicity',
                                                  'Income_Bracket', 'Education_Level',
                                                  'Health_Insurance_Status', 'Tract_Total_Pop',
                                                  'Median_Household_Income']]
            
            synthetic_individuals.append(tract_individuals)
        
        return pd.concat(synthetic_individuals, ignore_index=True) if synthetic_individuals else pd.DataFrame()

    # ============================================================================
    # STAGE 2: INDEPENDENT SCREENING STATUS ASSIGNMENT (with screening joint dist)
    # ============================================================================

    def is_eligible_age_group(self, age_group: str) -> bool:
        """Check if age group is eligible for screening (45-75 years)."""
        eligible_ages = ['45to49', '50to54', '55to59', '60to61', '62to64', '65to66',
                        '67to69', '70to74', '75to79']
        return age_group in eligible_ages

    def get_screening_joint_factors(self, geoid: str, age_group: str, race_ethnicity: str,
                                   insurance_status: str) -> Tuple[float, float, float]:
        """Extract screening adjustment factors from screening joint distributions."""
        if geoid not in self.screening_joint_dict:
            # Fallback to defaults if GEOID not in screening joint dist
            return 1.0, 1.0, 1.0
        
        row_data = self.screening_joint_dict[geoid]
        
        # Age adjustment
        age_col = f'{age_group}_Screening_Adjustment'
        age_adj = row_data.get(age_col, 1.0)
        if pd.isna(age_adj):
            age_adj = 1.0
        
        # Insurance adjustment
        insurance_col = f'{insurance_status}_Screening_Adjustment'
        insurance_adj = row_data.get(insurance_col, 1.0)
        if pd.isna(insurance_adj):
            insurance_adj = 1.0
        
        # Race adjustment
        race_col = f'{race_ethnicity}_Screening_Adjustment'
        race_adj = row_data.get(race_col, 1.0)
        if pd.isna(race_adj):
            race_adj = 1.0
        
        return float(age_adj), float(insurance_adj), float(race_adj)

    def calculate_screening_probability_with_joint_dist(self, tract_rate: float, 
                                                        geoid: str, age_group: str,
                                                        race_ethnicity: str,
                                                        insurance_status: str) -> float:
        """
        Calculate individual screening probability using screening joint distributions.
        INDEPENDENT of risk calculation.
        """
        
        age_adj, insurance_adj, race_adj = self.get_screening_joint_factors(
            geoid, age_group, race_ethnicity, insurance_status
        )
        
        combined_adjustment = age_adj * insurance_adj * race_adj
        adjusted_prob = tract_rate * combined_adjustment
        adjusted_prob = max(0.01, min(0.99, adjusted_prob))
        
        return adjusted_prob

    def assign_screening_status(self, population_df: pd.DataFrame) -> pd.DataFrame:
        """Assign colon cancer screening status using screening joint distributions (INDEPENDENT)."""
        screening_probs = []
        screening_status = []
        age_eligibility = []
        
        for idx, row in population_df.iterrows():
            age_group = row['Age_Group']
            
            if not self.is_eligible_age_group(age_group):
                screening_probs.append(0.0)
                screening_status.append('Not_Screened')
                age_eligibility.append('Outside_45_75_range')
            else:
                geoid = row['Tract_GEOID']
                tract_rate = self.colon_rates_dict.get(geoid, 0.68) / 100
                
                prob = self.calculate_screening_probability_with_joint_dist(
                    tract_rate=tract_rate,
                    geoid=geoid,
                    age_group=age_group,
                    race_ethnicity=row['Race_Ethnicity'],
                    insurance_status=row['Health_Insurance_Status']
                )
                
                screens = np.random.random() < prob
                screening_probs.append(prob)
                screening_status.append('Screened' if screens else 'Not_Screened')
                age_eligibility.append('Eligible_45_75')
        
        population_df['Age_Eligibility'] = age_eligibility
        population_df['Colon_Screening_Probability'] = screening_probs
        population_df['Colon_Cancer_Screening_Status'] = screening_status
        
        return population_df

    # ============================================================================
    # STAGE 3: INDEPENDENT CCRAT RISK ASSESSMENT
    # ============================================================================

    def map_age_group(self, age_group: str) -> Optional[str]:
        """Map Age_Group to CCRAT age category."""
        age_mapping = {
            'Under5': None, '5to9': None, '10to14': None, '15to17': None, '18to19': None,
            '20': None, '21': None, '22to24': None,
            '25to29': None, '30to34': None, '35to39': None, '40to44': None,
            '45to49': '45to49', '50to54': '50to54', '55to59': '55to59', '60to64': '60to64',
            '65to66': '65to69', '67to69': '65to69', '70to74': '70to75',
            '75to79': '70to75', '80to84': '70to75', '85plus': '70to75'
        }
        return age_mapping.get(age_group, None)

    def map_gender(self, gender: str) -> str:
        """Map Gender to CCRAT format."""
        gender_mapping = {'M': 'male', 'F': 'female'}
        return gender_mapping.get(gender, 'male')

    def map_race(self, race_ethnicity: str) -> str:
        """Map Race_Ethnicity to CCRAT format."""
        race_mapping = {
            'White_NonHispanic': 'white', 'Black_NonHispanic': 'black',
            'Hispanic_Latino': 'hispanic', 'Asian_NonHispanic': 'asian',
            'AIAN_NonHispanic': 'other', 'NHOPI_NonHispanic': 'other',
            'SomeOther_NonHispanic': 'other', 'TwoOrMore_NonHispanic': 'other'
        }
        return race_mapping.get(race_ethnicity, 'other')

    def map_income(self, income_bracket: str) -> str:
        """Map Income_Bracket to CCRAT format."""
        income_mapping = {
            'Less10k': 'under20k', '10to15k': 'under20k', '15to20k': 'under20k',
            '20to25k': '20kto35k', '25to30k': '20kto35k', '30to35k': '20kto35k',
            '35to40k': '35kto50k', '40to45k': '35kto50k', '45to50k': '35kto50k',
            '50to60k': '50kto75k', '60to75k': '50kto75k',
            '75to100k': 'over75k', '100to125k': 'over75k', '125to150k': 'over75k',
            '150to200k': 'over75k', '200kplus': 'over75k'
        }
        return income_mapping.get(income_bracket, '50kto75k')

    def map_education(self, education_level: str) -> str:
        """Map Education_Level to CCRAT format."""
        education_mapping = {
            'Under25': 'somecollege',
            'Less_than_9th_Grade': 'lessthanhs',
            '9th_to_12th_Grade_No_Diploma': 'lessthanhs',
            'High_School_Graduate': 'hsgraduate',
            'Some_College_No_Degree': 'somecollege',
            'Associates_Degree': 'somecollege',
            'Bachelors_Degree': 'bachelorplus',
            'Masters_Degree': 'bachelorplus',
            'Professional_Beyond_Masters': 'bachelorplus'
        }
        return education_mapping.get(education_level, 'hsgraduate')

    def calculate_risk(self, age_group: str, gender: str, race_ethnicity: str,
                       income_bracket: str, education_level: str) -> Optional[Dict]:
        """
        Calculate CCRAT risk for individual (INDEPENDENT of screening status).
        Always returns both screened and unscreened risk.

        Wrapper around the CSV-driven _calculate_ccrat_risk().
        """
        ccrat_age = self.map_age_group(age_group)
        if ccrat_age is None:
            return None

        ccrat_gender = self.map_gender(gender)
        ccrat_race = self.map_race(race_ethnicity)
        ccrat_income = self.map_income(income_bracket)
        ccrat_education = self.map_education(education_level)

        # Delegate to CSV-based implementation
        ccrat = self._calculate_ccrat_risk(
            age_group=age_group,
            gender=gender,
            race=race_ethnicity,
            income=income_bracket,
            education=education_level
        )
        if ccrat is None:
            return None

        # Adapt field names back to existing output schema
        return {
            'Unscreened_Risk': ccrat['UnscreenedRisk'],
            'Screened_Risk': ccrat['ScreenedRisk'],
            'Screening_Benefit': ccrat['ScreeningBenefit'],
            'Baseline_Risk': ccrat['BaselineRisk'],
            'Gender_Multiplier': ccrat['GenderMultiplier'],
            'Race_Multiplier': ccrat['RaceMultiplier'],
            'Income_Multiplier': ccrat['IncomeMultiplier'],
            'Education_Multiplier': ccrat['EducationMultiplier'],
            'Risk_Category': ccrat['RiskCategory'],
        }

    def assess_risk(self, population_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate CCRAT risk for entire population (INDEPENDENT)."""
        risks = []
        
        for idx, row in population_df.iterrows():
            risk = self.calculate_risk(
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
        
        return results

    # ============================================================================
    # MAIN PIPELINE EXECUTION
    # ============================================================================

    def _parse_ccrat_parameters(self, df: pd.DataFrame) -> dict:
        """Parse CCRAT parameters CSV into dictionaries for easy lookup."""
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

    def _categorize_risk(self, risk: float) -> str:
        """Categorize risk as Low, Medium, or High."""
        if risk < 0.4:
            return 'Low'
        elif risk < 0.8:
            return 'Medium'
        else:
            return 'High'

    def _calculate_ccrat_risk(self, age_group: str, gender: str, race: str,
                             income: str, education: str) -> Optional[Dict]:
        """Calculate CCRAT risk using parameters from CSV."""
        age_mapping = {
            '45to49': '45to49', '50to54': '50to54', '55to59': '55to59',
            '60to64': '60to64', '65to66': '65to69', '67to69': '65to69',
            '70to74': '70to75', '75to79': '70to75', '80to84': '70to75', '85plus': '70to75'
        }

        ccrat_age = age_mapping.get(age_group)
        if ccrat_age is None:
            return None

        baseline_risk = self.ccrat_parameters['age_baseline_risk'].get(ccrat_age, 0.5)
        gender_map = {'M': 'Male', 'F': 'Female'}
        gender_mult = self.ccrat_parameters['gender_multiplier'].get(gender_map.get(gender, 'Male'), 1.0)

        race_mapping = {
            'White_NonHispanic': 'White', 'Black_NonHispanic': 'Black',
            'Hispanic_Latino': 'Hispanic', 'Asian_NonHispanic': 'Asian',
            'AIAN_NonHispanic': 'Aian', 'NHOPI_NonHispanic': 'Nhopi',
            'SomeOther_NonHispanic': 'Someother', 'TwoOrMore_NonHispanic': 'Twomore'
        }
        race_mult = self.ccrat_parameters['race_multiplier'].get(race_mapping.get(race, 'White'), 1.0)

        income_mapping = {
            'Less10k': 'under20k', '10to15k': 'under20k', '15to20k': 'under20k',
            '20to25k': '20kto35k', '25to30k': '20kto35k', '30to35k': '20kto35k',
            '35to40k': '35kto50k', '40to45k': '35kto50k', '45to50k': '35kto50k',
            '50to60k': '50kto75k', '60to75k': '50kto75k',
            '75to100k': 'over75k', '100to125k': 'over75k', '125to150k': 'over75k',
            '150to200k': 'over75k', '200kplus': 'over75k'
        }
        income_mult = self.ccrat_parameters['income_multiplier'].get(income_mapping.get(income, '50kto75k'), 1.0)

        education_mapping = {
            'Under25': 'somecollege', 'Less_than_9th_Grade': 'lessthanhs',
            '9th_to_12th_Grade_No_Diploma': 'lessthanhs', 'High_School_Graduate': 'hsgraduate',
            'Some_College_No_Degree': 'somecollege', 'Associates_Degree': 'somecollege',
            'Bachelors_Degree': 'bachelorplus', 'Masters_Degree': 'bachelorplus',
            'Professional_Beyond_Masters': 'bachelorplus'
        }
        education_mult = self.ccrat_parameters['education_multiplier'].get(education_mapping.get(education, 'hsgraduate'), 1.0)

        combined_multiplier = gender_mult * race_mult * income_mult * education_mult
        unscreened_risk = baseline_risk * combined_multiplier
        screening_effectiveness = self.ccrat_parameters['screening_effectiveness']
        screened_risk = unscreened_risk * (1 - screening_effectiveness)

        return {
            'UnscreenedRisk': unscreened_risk,
            'ScreenedRisk': screened_risk,
            'ScreeningBenefit': unscreened_risk - screened_risk,
            'BaselineRisk': baseline_risk,
            'GenderMultiplier': gender_mult,
            'RaceMultiplier': race_mult,
            'IncomeMultiplier': income_mult,
            'EducationMultiplier': education_mult,
            'RiskCategory': self._categorize_risk(unscreened_risk)
        }

    def run(self, output_file: str, sample_size: int = 100) -> pd.DataFrame:
        """
        Execute the complete integrated pipeline with INDEPENDENT stages.
        
        Args:
            output_file: Path to save the final CSV with all attributes
            sample_size: Target number of individuals per tract (default: 100)
        
        Returns:
            DataFrame with complete synthetic population
        """
        
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: GENERATING SYNTHETIC POPULATION (IPF with IPF Joint Distributions)")
        logger.info("=" * 80)
        
        synth_pop = self.generate_synthetic_population(sample_size=sample_size)
        logger.info(f"✓ Generated {len(synth_pop)} synthetic individuals\n")
        
        logger.info("=" * 80)
        logger.info("STAGE 2: ASSIGNING SCREENING STATUS (INDEPENDENT - Using Screening Joint Distributions)")
        logger.info("=" * 80)
        
        with_screening = self.assign_screening_status(synth_pop.copy())
        screened = (with_screening['Colon_Cancer_Screening_Status'] == 'Screened').sum()
        logger.info(f"✓ Assigned screening status independently")
        logger.info(f"  - Screened: {screened} ({screened/len(with_screening)*100:.1f}%)")
        logger.info(f"  - Not screened: {len(with_screening) - screened} ({(len(with_screening) - screened)/len(with_screening)*100:.1f}%)\n")
        
        logger.info("=" * 80)
        logger.info("STAGE 3: CALCULATING RISK ASSESSMENT (INDEPENDENT)")
        logger.info("=" * 80)
        
        final_results = self.assess_risk(with_screening)
        logger.info(f"✓ Calculated risk independently\n")
        
        logger.info("=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80)
        
        final_results.to_csv(output_file, index=False)
        logger.info(f"✓ Saved complete synthetic population to: {output_file}\n")
        
        self._print_summary(final_results)
        
        return final_results

    def _print_summary(self, results: pd.DataFrame):
        """Print comprehensive summary statistics."""
        
        logger.info("=" * 80)
        logger.info("FINAL SUMMARY STATISTICS")
        logger.info("=" * 80)
        
        logger.info(f"\nPopulation Overview:")
        logger.info(f"  Total individuals: {len(results)}")
        logger.info(f"  Unique tracts: {results['Tract_GEOID'].nunique()}")
        
        logger.info(f"\nScreening Status (INDEPENDENT calculation with Screening Joint Distributions):")
        screened = (results['Colon_Cancer_Screening_Status'] == 'Screened').sum()
        not_screened = (results['Colon_Cancer_Screening_Status'] == 'Not_Screened').sum()
        logger.info(f"  Screened: {screened} ({screened/len(results)*100:.1f}%)")
        logger.info(f"  Not screened: {not_screened} ({not_screened/len(results)*100:.1f}%)")
        
        eligible_df = results[results['Age_Eligibility'] == 'Eligible_45_75']
        if len(eligible_df) > 0:
            logger.info(f"\n  Among eligible ages (45-75):")
            screened_eligible = (eligible_df['Colon_Cancer_Screening_Status'] == 'Screened').sum()
            logger.info(f"    Screened: {screened_eligible} ({screened_eligible/len(eligible_df)*100:.1f}%)")
        
        screening_df = results[results['Unscreened_Risk'].notna()]
        if len(screening_df) > 0:
            logger.info(f"\nRisk Distribution (INDEPENDENT calculation):")
            logger.info(f"  Mean unscreened risk: {screening_df['Unscreened_Risk'].mean():.4f}")
            logger.info(f"  Median unscreened risk: {screening_df['Unscreened_Risk'].median():.4f}")
            logger.info(f"  Std dev: {screening_df['Unscreened_Risk'].std():.4f}")
            
            logger.info(f"\nRisk Categories:")
            categories = screening_df['Risk_Category'].value_counts()
            for cat, count in categories.items():
                pct = (count / len(screening_df)) * 100
                logger.info(f"  {cat}: {count} ({pct:.1f}%)")
        
        logger.info(f"\n{'='*80}\n")

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Integrated Synthetic Population Pipeline")
    parser.add_argument("--demographics", required=True, help="Demographics CSV file")
    parser.add_argument("--ipf-joint-dist", required=True, help="IPF joint distributions CSV")
    parser.add_argument("--screening-joint-dist", required=True, help="Screening joint distributions CSV")
    parser.add_argument("--colon-rates", required=True, help="Colon screening rates CSV")
    parser.add_argument("--ccrat-parameters", required=True, help="CCRAT parameters CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--scaling-factor", type=int, default=100, help="Scaling factor (default: 100)")
    
    args = parser.parse_args()
    
    pipeline = IntegratedSyntheticPopulationPipeline(
        demographics_csv=args.demographics,
        ipf_joint_distributions_csv=args.ipf_joint_dist,
        screening_joint_distributions_csv=args.screening_joint_dist,
        colon_rates_csv=args.colon_rates,
        ccrat_parameters_csv=args.ccrat_parameters,
        scaling_factor=args.scaling_factor
    )
    
    results = pipeline.run(output_file=args.output)
