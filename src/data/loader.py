"""
Data loading utilities with schema validation.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class EnzymeDataLoader:
    """Load and validate enzyme datasets."""
    
    REQUIRED_COLUMNS = [
        'ec', 'organism', 'log_kcat', 'n_measurements', 
        'uniprot_primary', 'sequence'
    ]
    
    OPTIONAL_COLUMNS = [
        'ph_opt', 'ph_max', 'temp_opt', 'temp_max',
        'molecularWeight', 'kmValue', 'kcat_std'
    ]
    
    def __init__(self, config: Dict):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration dictionary from agent.yaml
        """
        self.config = config
        self.datasets = config['datasets']['enzymes']
        
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load a single enzyme dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'gst', 'laccase', 'lactase')
            
        Returns:
            DataFrame with validated schema
        """
        # Find dataset config
        dataset_config = None
        for ds in self.datasets:
            if ds['name'] == dataset_name:
                dataset_config = ds
                break
                
        if dataset_config is None:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        
        csv_path = Path(dataset_config['csv'])
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        
        logger.info(f"Loading dataset: {dataset_name} from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Validate schema
        self._validate_schema(df, dataset_name)
        
        # Add metadata
        df['dataset_name'] = dataset_name
        df['ec_number'] = dataset_config['ec']
        df['real_world_problem'] = dataset_config['real_world_problem']
        
        logger.info(f"Loaded {len(df)} records for {dataset_name}")
        return df
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all configured datasets.
        
        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        datasets = {}
        for ds_config in self.datasets:
            name = ds_config['name']
            datasets[name] = self.load_dataset(name)
        return datasets
    
    def _validate_schema(self, df: pd.DataFrame, dataset_name: str):
        """Validate that DataFrame has required columns."""
        missing_cols = []
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(
                f"Dataset '{dataset_name}' missing required columns: {missing_cols}"
            )
        
        # Log optional columns status
        present_optional = [col for col in self.OPTIONAL_COLUMNS if col in df.columns]
        missing_optional = [col for col in self.OPTIONAL_COLUMNS if col not in df.columns]
        
        if present_optional:
            logger.info(f"Optional columns present: {present_optional}")
        if missing_optional:
            logger.info(f"Optional columns missing (will be imputed): {missing_optional}")
