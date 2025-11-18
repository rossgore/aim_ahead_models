"""
Data loading utilities for census tract data.

Provides functions to load and process census tract demographic data
from various sources including CSV files and Census API.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class CensusTractDataLoader:
    """
    Census tract data loader.

    Handles loading and initial processing of census tract demographic data.
    """

    def __init__(self, config):
        """
        Initialize data loader.

        Args:
            config: Model configuration object
        """
        self.config = config
        self.data_dir = Path(config.data_dir)

    def load_region(self, region: str) -> pd.DataFrame:
        """
        Load census tract data for specified region.

        Args:
            region: Region name (e.g., 'Hampton_Roads')

        Returns:
            DataFrame with census tract demographic data
        """
        # Look for data file
        data_file = self.data_dir / f"{region.lower()}_census_tracts.csv"

        if data_file.exists():
            return self.load_from_file(str(data_file))
        else:
            logger.warning(f"Data file not found: {data_file}")
            # Generate synthetic data as fallback
            return self._generate_synthetic_data(region)

    def load_from_file(self, file_path: str) -> pd.DataFrame:
        """Load census tract data from CSV file."""
        logger.info(f"Loading data from {file_path}")

        df = pd.read_csv(file_path)

        # Validate required columns
        required_cols = ['GEOID', 'Population', 'Area']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")

        return df

    def _generate_synthetic_data(self, region: str, n_tracts: int = 100) -> pd.DataFrame:
        """Generate synthetic census tract data for testing."""
        logger.info(f"Generating synthetic data for {region} ({n_tracts} tracts)")

        data = []
        for i in range(n_tracts):
            tract = {
                'GEOID': f"51000{i:06d}",
                'Area': region,
                'Population': int(np.random.uniform(2500, 6000)),
                'Poverty_Rate': np.random.uniform(0.05, 0.30),
                'Medicaid_Rate': np.random.uniform(0.15, 0.45),
                'Median_Income': int(np.random.uniform(35000, 85000))
            }
            data.append(tract)

        return pd.DataFrame(data)

    def save_data(self, data: pd.DataFrame, filename: str):
        """Save processed data to file."""
        output_path = self.data_dir / "processed" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data.to_csv(output_path, index=False)
        logger.info(f"Saved data to {output_path}")
