"""
Configuration management for CCRAT model.

Centralizes model configuration parameters and provides validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for CCRAT-Enhanced Model.

    Attributes:
        version: Model version
        medicaid_loss_rate: Base Medicaid coverage loss rate
        time_horizon: Projection time horizon in years
        discount_rate: Annual discount rate for cost calculations
        random_seed: Random seed for reproducibility
    """
    version: str = "4.0.0"
    medicaid_loss_rate: float = 0.146
    time_horizon: int = 10
    discount_rate: float = 0.03
    random_seed: Optional[int] = None

    # Data paths
    data_dir: str = "data"
    output_dir: str = "outputs"

    # Validation settings
    validation_tolerance: float = 0.10
    enable_validation: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModelConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'version': self.version,
            'medicaid_loss_rate': self.medicaid_loss_rate,
            'time_horizon': self.time_horizon,
            'discount_rate': self.discount_rate,
            'random_seed': self.random_seed,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'validation_tolerance': self.validation_tolerance,
            'enable_validation': self.enable_validation
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def validate(self) -> bool:
        """Validate configuration parameters."""
        checks = []

        # Check rates are between 0 and 1
        checks.append(0 <= self.medicaid_loss_rate <= 1)
        checks.append(0 <= self.discount_rate <= 1)
        checks.append(0 <= self.validation_tolerance <= 1)

        # Check time horizon is positive
        checks.append(self.time_horizon > 0)

        is_valid = all(checks)
        if not is_valid:
            logger.error("Invalid configuration parameters")

        return is_valid
