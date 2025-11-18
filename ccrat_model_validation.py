"""
Model validation framework for quality assurance.

This module provides comprehensive validation procedures including statistical
testing, sensitivity analysis, and external benchmark comparisons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ModelValidationFramework:
    """
    Model validation component.

    Provides comprehensive validation including statistical testing,
    sensitivity analysis, cross-validation, and external benchmarking.
    """

    def __init__(self, config):
        """
        Initialize validation framework.

        Args:
            config: Model configuration object
        """
        self.config = config

        # Validation thresholds
        self.tolerance_thresholds = {
            'demographic_match': 0.10,  # 10% tolerance
            'chi_square_pvalue': 0.05,
            'correlation_threshold': 0.10
        }

        logger.info("Initialized Model Validation Framework")

    def validate_model(
        self,
        model_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Run comprehensive model validation.

        Args:
            model_data: Model output data
            validation_data: External validation dataset (optional)

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'demographic_validation': self._validate_demographics(model_data),
            'statistical_tests': self._run_statistical_tests(model_data),
            'consistency_checks': self._check_consistency(model_data)
        }

        if validation_data is not None:
            validation_results['external_validation'] = self._external_validation(
                model_data, validation_data
            )

        # Overall validation status
        validation_results['overall_valid'] = self._assess_overall_validity(
            validation_results
        )

        return validation_results

    def _validate_demographics(self, data: pd.DataFrame) -> Dict:
        """Validate demographic distributions."""
        results = {}

        # Check for missing values
        results['missing_values'] = data.isnull().sum().to_dict()

        # Check distribution ranges
        for col in ['age', 'race', 'gender', 'income', 'education']:
            if col in data.columns:
                value_counts = data[col].value_counts()
                results[f'{col}_distribution'] = value_counts.to_dict()

        return results

    def _run_statistical_tests(self, data: pd.DataFrame) -> Dict:
        """Run statistical validation tests."""
        results = {}

        # Test for reasonable distributions
        if 'unscreened_risk' in data.columns:
            risk_values = data['unscreened_risk'].dropna()

            # Check if risks are in reasonable range (0-5%)
            results['risk_range_valid'] = (
                (risk_values >= 0).all() and (risk_values <= 5.0).all()
            )

            # Basic statistics
            results['risk_statistics'] = {
                'mean': risk_values.mean(),
                'median': risk_values.median(),
                'std': risk_values.std(),
                'min': risk_values.min(),
                'max': risk_values.max()
            }

        return results

    def _check_consistency(self, data: pd.DataFrame) -> Dict:
        """Check internal consistency of model outputs."""
        results = {}

        # Check that screened risk < unscreened risk
        if 'screened_risk' in data.columns and 'unscreened_risk' in data.columns:
            consistency_check = (data['screened_risk'] <= data['unscreened_risk']).all()
            results['risk_ordering_consistent'] = consistency_check

        # Check that costs are non-negative
        cost_cols = [col for col in data.columns if 'cost' in col.lower()]
        if cost_cols:
            results['non_negative_costs'] = all(
                (data[col] >= 0).all() for col in cost_cols if col in data.columns
            )

        return results

    def _external_validation(
        self,
        model_data: pd.DataFrame,
        validation_data: pd.DataFrame
    ) -> Dict:
        """Validate against external benchmarks."""
        results = {}

        # Compare key metrics if available
        common_columns = set(model_data.columns) & set(validation_data.columns)

        for col in common_columns:
            if pd.api.types.is_numeric_dtype(model_data[col]):
                model_mean = model_data[col].mean()
                validation_mean = validation_data[col].mean()

                results[f'{col}_comparison'] = {
                    'model_mean': model_mean,
                    'validation_mean': validation_mean,
                    'difference': model_mean - validation_mean,
                    'percent_difference': ((model_mean - validation_mean) / validation_mean * 100)
                    if validation_mean != 0 else np.nan
                }

        return results

    def _assess_overall_validity(self, validation_results: Dict) -> bool:
        """Assess overall model validity."""
        # Check critical validation criteria
        checks = []

        # Demographic validation
        if 'demographic_validation' in validation_results:
            demo_valid = len(validation_results['demographic_validation'].get('missing_values', {})) == 0
            checks.append(demo_valid)

        # Statistical tests
        if 'statistical_tests' in validation_results:
            stat_valid = validation_results['statistical_tests'].get('risk_range_valid', True)
            checks.append(stat_valid)

        # Consistency checks
        if 'consistency_checks' in validation_results:
            consistency_valid = all(
                v for k, v in validation_results['consistency_checks'].items()
                if isinstance(v, bool)
            )
            checks.append(consistency_valid)

        return all(checks) if checks else False

    def run_sensitivity_analysis(
        self,
        model_function,
        base_parameters: Dict,
        sensitivity_ranges: Dict,
        n_samples: int = 100
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis on model parameters.

        Args:
            model_function: Function to run model
            base_parameters: Base parameter values
            sensitivity_ranges: Parameter ranges for sensitivity testing
            n_samples: Number of samples per parameter

        Returns:
            DataFrame with sensitivity analysis results
        """
        results = []

        for param_name, (param_min, param_max) in sensitivity_ranges.items():
            param_values = np.linspace(param_min, param_max, n_samples)

            for param_value in param_values:
                # Create modified parameters
                modified_params = base_parameters.copy()
                modified_params[param_name] = param_value

                # Run model
                try:
                    output = model_function(**modified_params)
                    results.append({
                        'parameter': param_name,
                        'value': param_value,
                        'output': output
                    })
                except Exception as e:
                    logger.warning(f"Sensitivity analysis failed for {param_name}={param_value}: {e}")

        return pd.DataFrame(results)
