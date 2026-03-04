import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ModalityAssigner:
    """
    Assigns a specific screening modality to individuals already flagged as 'Screened'.
    Does not modify risk, cost, or eligibility calculations (Milestone 1).
    """

    def __init__(self, cancer_type: str, screening_modalities_csv: str):
        # 1. Store and validate cancer type (Matching Design Doc)
        if cancer_type not in ['colon', 'breast']:
            raise ValueError("cancer_type must be 'colon' or 'breast'")
        self.cancer_type = cancer_type
        
        # 2. Load modalities CSV
        try:
            modalities_df = pd.read_csv(screening_modalities_csv)
        except FileNotFoundError:
            raise ValueError(f"Could not find the modalities file at: {screening_modalities_csv}")
        
        # 3. Robust Column Mapping (Handles both your actual CSV and the Design Doc spec)
        column_mapping = {
            'Modality': 'modality',
            'Availability': 'availability_weight',
            'Uptake': 'patient_uptake',
            'Interval (yrs)': 'interval_years',
            'Cost': 'cost_usd'
        }
        modalities_df = modalities_df.rename(columns=column_mapping)
        
        # 4. Filter by cancer type (Only if the column exists in the CSV)
        if 'cancer_type' in modalities_df.columns:
            modalities_df = modalities_df[modalities_df['cancer_type'] == self.cancer_type].copy()
            
        # 5. Validate required columns exist for MS1 math
        required_columns = ['modality', 'availability_weight', 'patient_uptake']
        for col in required_columns:
            if col not in modalities_df.columns:
                raise ValueError(f"CSV is missing required column: {col}")
                
        # 6. Compute combined weight (Availability * Uptake)
        modalities_df['combined_weight'] = modalities_df['availability_weight'] * modalities_df['patient_uptake']
        total_weight = modalities_df['combined_weight'].sum()
        
        if total_weight == 0:
            raise ValueError("All combined weights are zero <- check CSV values")
            
        # Store normalized weights and names for probability sampling
        self.normalized_weights = (modalities_df['combined_weight'] / total_weight).tolist()
        self.modality_names = modalities_df['modality'].tolist()

    def assign_modality_to_population(self, population_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies screened individuals and assigns a modality based on probability weights.
        """
        if 'Colon_Cancer_Screening_Status' not in population_df.columns:
            raise ValueError("Run Screening Calculator before ModalityAssigner")
            
        screened_mask = population_df['Colon_Cancer_Screening_Status'] == 'Screened'
        screened_df = population_df[screened_mask].copy()
        unscreened_df = population_df[~screened_mask].copy()
        
        if len(screened_df) == 0:
            logger.warning("No screened individuals found <- returning population unchanged")
            result_df = population_df.copy()
            result_df['Screening_Modality'] = 'None'
            return result_df
            
        unscreened_df['Screening_Modality'] = 'None'
        
        n_screened = len(screened_df)
        assigned = np.random.choice(
            self.modality_names, 
            size=n_screened, 
            p=self.normalized_weights
        )
        screened_df['Screening_Modality'] = assigned
        
        # Recombine and sort back to original index to preserve order for Stage 3
        result_df = pd.concat([screened_df, unscreened_df])
        result_df = result_df.loc[population_df.index]
        
        # 7. Summary Logging (Addressing the critique)
        logger.info(f"✓ Assigned {self.cancer_type} screening modalities:")
        modality_counts = result_df['Screening_Modality'].value_counts()
        total_pop = len(result_df)
        for mod, count in modality_counts.items():
            logger.info(f"  - {mod}: {count} ({count/total_pop*100:.1f}% of total pop)")
            
        return result_df