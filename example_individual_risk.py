"""
Individual risk calculation example.

Demonstrates CCRAT-style individual colorectal cancer risk assessment
for different demographic profiles.
"""

from ccrat_model import CCRATEnhancedModel
import pandas as pd


def main():
    """Calculate and compare individual risks."""

    print("CCRAT Individual Risk Assessment Examples")
    print("=" * 70)

    # Initialize model
    model = CCRATEnhancedModel()

    # Define example individuals
    individuals = [
        {
            'name': 'Affluent White Male',
            'age': '50_54',
            'gender': 'male',
            'race': 'white',
            'income': 'over_75k',
            'education': 'bachelor_plus'
        },
        {
            'name': 'Disadvantaged Black Female',
            'age': '60_64',
            'gender': 'female',
            'race': 'black',
            'income': 'under_20k',
            'education': 'less_than_hs'
        },
        {
            'name': 'Hispanic Working Class Male',
            'age': '65_69',
            'gender': 'male',
            'race': 'hispanic',
            'income': '35k_50k',
            'education': 'hs_graduate'
        },
        {
            'name': 'Asian Middle Income Female',
            'age': '55_59',
            'gender': 'female',
            'race': 'asian',
            'income': '50k_75k',
            'education': 'some_college'
        }
    ]

    # Calculate risk for each individual
    results = []
    for person in individuals:
        risk = model.calculate_individual_risk(
            age=person['age'],
            gender=person['gender'],
            race=person['race'],
            income=person['income'],
            education=person['education']
        )

        results.append({
            'name': person['name'],
            'unscreened_risk': risk['unscreened_risk'],
            'screened_risk': risk['screened_risk'],
            'screening_benefit': risk['screening_benefit']
        })

        # Display individual results
        print(f"\n{person['name']}:")
        print(f"  Age: {person['age'].replace('_', '-')}, Gender: {person['gender'].capitalize()}")
        print(f"  Race: {person['race'].capitalize()}, Income: {person['income']}")
        print(f"  Education: {person['education'].replace('_', ' ')}")
        print(f"  \n  5-year cancer risk (unscreened): {risk['unscreened_risk']:.3f}%")
        print(f"  5-year cancer risk (screened): {risk['screened_risk']:.3f}%")
        print(f"  Screening benefit: {risk['screening_benefit']:.3f}%")
        print(f"  \n  Risk multipliers:")
        print(f"    - Gender: {risk['risk_components']['gender_multiplier']:.2f}x")
        print(f"    - Race: {risk['risk_components']['race_multiplier']:.2f}x")
        print(f"    - Income: {risk['risk_components']['income_multiplier']:.2f}x")
        print(f"    - Education: {risk['risk_components']['education_multiplier']:.2f}x")

    # Summary comparison
    print("\n" + "=" * 70)
    print("RISK COMPARISON SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print(f"\nRisk Range: {results_df['unscreened_risk'].min():.3f}% to {results_df['unscreened_risk'].max():.3f}%")
    print(f"Risk Variation: {results_df['unscreened_risk'].max() / results_df['unscreened_risk'].min():.1f}x")
    print(f"Mean Screening Benefit: {results_df['screening_benefit'].mean():.3f}%")


if __name__ == "__main__":
    main()
