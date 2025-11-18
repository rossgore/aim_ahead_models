"""
Insurance coverage transition submodel for modeling Medicaid loss.

This module models the transition from Medicaid coverage to uninsured status
under policy changes, incorporating demographic variation in coverage vulnerability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class InsuranceCoverageTransition:
    """
    Insurance coverage transition component.

    Models probability of Medicaid coverage loss and transitions to uninsured
    status under policy changes, with demographic-specific adjustment factors.
    """

    def __init__(self, config):
        """
        Initialize insurance coverage transition component.

        Args:
            config: Model configuration object
        """
        self.config = config

        # Base Medicaid loss rate
        self.medicaid_loss_rate = 0.146  # 14.6% baseline

        # Demographic adjustment factors (placeholder - would be calibrated)
        self.demographic_loss_adjustments = {
            'age': {
                '45_49': 1.10, '50_54': 1.05, '55_59': 1.00,
                '60_64': 0.95, '65_69': 0.80, '70_75': 0.75
            },
            'race': {
                'white': 1.00, 'black': 1.05, 'hispanic': 1.10,
                'asian': 0.95, 'other': 1.00
            }
        }

        logger.info("Initialized Insurance Coverage Transition component")

    def model_transitions(
        self,
        population: pd.DataFrame,
        medicaid_rate: Optional[float] = None
    ) -> Dict:
        """
        Model insurance coverage transitions for population.

        Args:
            population: DataFrame with individual demographic data
            medicaid_rate: Tract-level Medicaid enrollment rate (optional)

        Returns:
            Dictionary with transition statistics
        """
        # Identify Medicaid enrollees
        if 'medicaid_enrolled' not in population.columns:
            # Estimate Medicaid enrollment based on income
            population = population.copy()
            medicaid_prob = population['income'].map({
                'under_20k': 0.45,
                '20k_35k': 0.30,
                '35k_50k': 0.15,
                '50k_75k': 0.05,
                'over_75k': 0.02
            })
            population['medicaid_enrolled'] = np.random.random(len(population)) < medicaid_prob

        medicaid_enrollees = population[population['medicaid_enrolled']]

        # Calculate loss probability for each enrollee
        loss_probs = []
        for _, individual in medicaid_enrollees.iterrows():
            loss_prob = self._calculate_loss_probability(individual)
            loss_probs.append(loss_prob)

        medicaid_enrollees = medicaid_enrollees.copy()
        medicaid_enrollees['loss_probability'] = loss_probs
        medicaid_enrollees['loses_coverage'] = (
            np.random.random(len(medicaid_enrollees)) < medicaid_enrollees['loss_probability']
        )

        # Calculate statistics
        total_medicaid = len(medicaid_enrollees)
        total_losses = medicaid_enrollees['loses_coverage'].sum()

        return {
            'total_medicaid_enrollees': total_medicaid,
            'medicaid_losses': total_losses,
            'loss_rate': total_losses / total_medicaid if total_medicaid > 0 else 0,
            'population_with_transitions': medicaid_enrollees
        }

    def _calculate_loss_probability(self, individual: pd.Series) -> float:
        """Calculate Medicaid loss probability for individual."""
        base_prob = self.medicaid_loss_rate

        # Apply demographic adjustments
        age_adj = self.demographic_loss_adjustments['age'].get(individual['age'], 1.0)
        race_adj = self.demographic_loss_adjustments['race'].get(individual['race'], 1.0)

        adjusted_prob = base_prob * age_adj * race_adj

        return min(adjusted_prob, 0.95)  # Cap at 95%
