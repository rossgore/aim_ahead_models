import pandas as pd
from cancer_economics_model import CancerEconomicsModel

def test_economics_model():
    print("Loading synthetic population...")
    # Point this to the output file you generated with model.py
    population_df = pd.read_csv('output/synthetic_population_colon.csv')
    
    print("Initializing CancerEconomicsModel...")
    # Make sure these paths match where your CSVs are actually located
    model = CancerEconomicsModel(
        cancer_type='colon',
        parameters_csv='data/colon_cancer_economics_parameters.csv', 
        screening_modalities_csv='data/Modality-Sensitivity-Availability-Uptake-Intervalyrs-Cost.csv'
    )
    
    print("Applying costs to population...")
    costs_df = model.apply_costs_to_population(population_df)
    
    print("Generating cost report...")
    report = model.generate_cost_report(costs_df)
    
    print("\n--- SUCCESS! Cost Report Summary ---")
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key}: {value:,.2f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    test_economics_model()