"""
Health economics submodel for projecting healthcare costs.

This module projects long-term healthcare costs based on screening patterns
and cancer stage distribution using stage-specific treatment cost data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class HealthEconomicsModel:
    """
    Health economics component.

    Projects long-term healthcare costs based on cancer stage distribution
    changes resulting from screening pattern modifications.
    """

    def __init__(self, config):
        """
        Initialize health economics component.

        Args:
            config: Model configuration object
        """
        self.config = config

        # Treatment costs by cancer stage (from SEER-Medicare data)
        self.treatment_costs = {
            'Stage_I': 31063,
            'Stage_II': 39834,
            'Stage_III': 45000,
            'Stage_IV': 108599
        }

        # Stage distributions
        self.stage_distributions = {
            'screened': {
                'Stage_I': 0.42,
                'Stage_II': 0.28,
                'Stage_III': 0.21,
                'Stage_IV': 0.09
            },
            'unscreened': {
                'Stage_I': 0.18,
                'Stage_II': 0.22,
                'Stage_III': 0.35,
                'Stage_IV': 0.25
            }
        }

        # Model parameters
        self.time_horizon = 10  # years
        self.discount_rate = 0.03

        logger.info("Initialized Health Economics Model component")

    def calculate_costs(
        self,
        population: pd.DataFrame,
        risks: pd.DataFrame,
        screening: Dict
    ) -> Dict:
        """
        Calculate healthcare costs for population.

        Args:
            population: DataFrame with demographic data
            risks: DataFrame with individual risk estimates
            screening: Dictionary with screening utilization predictions

        Returns:
            Dictionary with cost projections
        """
        pop_with_screening = screening['population_with_screening']

        # Calculate costs for each individual
        costs_before = []
        costs_after = []

        for idx, individual in pop_with_screening.iterrows():
            # Find corresponding risk
            if idx < len(risks):
                individual_risk = risks.iloc[idx]['unscreened_risk'] / 100  # Convert to decimal
            else:
                individual_risk = 0.005  # Default risk

            # Cost with Medicaid (high screening rate)
            screening_rate_before = individual['screening_rate_before']
            cost_before = self._calculate_expected_cost(
                individual_risk,
                screening_rate_before
            )
            costs_before.append(cost_before)

            # Cost after potential coverage loss
            screening_rate_after = individual['screening_rate_after']
            cost_after = self._calculate_expected_cost(
                individual_risk,
                screening_rate_after
            )
            costs_after.append(cost_after)

        pop_with_screening = pop_with_screening.copy()
        pop_with_screening['cost_before'] = costs_before
        pop_with_screening['cost_after'] = costs_after
        pop_with_screening['cost_increase'] = (
            pop_with_screening['cost_after'] - pop_with_screening['cost_before']
        )

        # Aggregate costs
        total_cost_increase = pop_with_screening['cost_increase'].sum() * self.time_horizon

        return {
            'total_impact': total_cost_increase,
            'avg_cost_per_person': total_cost_increase / len(pop_with_screening),
            'population_with_costs': pop_with_screening
        }

    def _calculate_expected_cost(
        self,
        cancer_risk: float,
        screening_rate: float
    ) -> float:
        """Calculate expected lifetime cancer treatment cost."""
        # Determine stage distribution based on screening status
        # Simplification: use weighted average of screened/unscreened distributions
        stage_dist = {}
        for stage in self.treatment_costs.keys():
            stage_dist[stage] = (
                screening_rate * self.stage_distributions['screened'][stage] +
                (1 - screening_rate) * self.stage_distributions['unscreened'][stage]
            )

        # Calculate expected cost
        expected_cost = 0
        for stage, probability in stage_dist.items():
            expected_cost += cancer_risk * probability * self.treatment_costs[stage]

        return expected_cost
