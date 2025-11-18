"""
Basic census tract analysis example.

This script demonstrates basic usage of the CCRAT-Enhanced Model
for analyzing census tract-level healthcare impacts.
"""

from ccrat_model import CCRATEnhancedModel
import pandas as pd


def main():
    """Run basic census tract analysis."""

    print("CCRAT-Enhanced Census Tract Model - Basic Analysis Example")
    print("=" * 60)

    # Initialize model
    print("\n1. Initializing model...")
    model = CCRATEnhancedModel()

    # Load census tract data (using synthetic data for demo)
    print("2. Loading census tract data...")
    model.load_census_tract_data(region='Hampton_Roads')

    # Analyze overall population impact
    print("3. Analyzing population impact...")
    results = model.analyze_population()

    # Display summary results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total affected population: {results['summary']['total_affected']:,.0f}")
    print(f"Total cost impact: ${results['summary']['total_cost']:,.0f}")
    print(f"Cost per affected person: ${results['summary']['cost_per_person']:,.0f}")

    # Analyze specific demographic group
    print("\n4. Analyzing Black population subgroup...")
    black_results = model.analyze_population(
        demographic_filters={'race': ['black']}
    )

    print(f"Black population affected: {black_results['summary']['total_affected']:,.0f}")
    print(f"Black population cost impact: ${black_results['summary']['total_cost']:,.0f}")

    # Identify high-risk tracts
    print("\n5. Identifying high-risk census tracts...")
    high_risk_tracts = model.identify_high_risk_tracts(top_n=5)

    print("\nTop 5 High-Risk Census Tracts:")
    for idx, tract in high_risk_tracts.iterrows():
        print(f"  Tract {tract['geoid']}: ${tract['cost_impact']:,.0f} impact")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
