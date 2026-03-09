# test_screening_calculator.py
import pandas as pd
from screening_calculator import ScreeningCalculator

# Create minimal test data
test_df = pd.DataFrame({
    'Tract_GEOID': ['1400000US51810045000'] * 4,
    'Age_Group': ['45to49', '55to59', '40to44', '70to74'],
    'Race_Ethnicity': ['White_NonHispanic'] * 4,
    'Health_Insurance_Status': ['Insured'] * 4
})

# Test colon calculator
print("Testing colon calculator...")
colon_calc = ScreeningCalculator(
    cancer_type='colon',
    screening_joint_distributions_csv='data/colon-screening-joint-distributions.csv',
    screening_rates_csv='data/colon-rates.csv'
)
colon_result = colon_calc.assign_screening_to_population(test_df)
print("✓ Colon columns:", [c for c in colon_result.columns if 'Colon' in c or 'Eligibility' in c])

# Test breast calculator
print("\nTesting breast calculator...")
breast_calc = ScreeningCalculator(
    cancer_type='breast',
    screening_joint_distributions_csv='data/breast-screening-joint-distributions.csv',
    screening_rates_csv='data/breast-cancer-rates.csv'
)
breast_result = breast_calc.assign_screening_to_population(test_df)
print("✓ Breast columns:", [c for c in breast_result.columns if 'Breast' in c or 'Eligibility' in c])

# Check age eligibility logic
print("\nAge eligibility checks:")
print("  Colon - Age 45to49:", colon_result.iloc[0]['Age_Eligibility'])
print("  Colon - Age 40to44:", colon_result.iloc[2]['Age_Eligibility'])
print("  Breast - Age 40to44:", breast_result.iloc[2]['Age_Eligibility'])
print("  Breast - Age 70to74:", breast_result.iloc[3]['Age_Eligibility'])

print("\n✓ Step 1 validation complete!")
