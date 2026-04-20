"""Data loading service for bioprocess optimization."""
import pandas as pd
from pathlib import Path
from typing import Dict
from app.utils import get_logger

logger = get_logger(__name__)


class DataService:
    """Service for loading enzyme datasets."""
    
    _instance = None
    _datasets: Dict[str, pd.DataFrame] = {}
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_datasets(self, data_dir: str = "data") -> None:
        """Load all enzyme datasets from CSV files.
        
        Args:
            data_dir: Directory containing CSV files
        """
        data_path = Path(data_dir)
        
        for dataset_name in ['gst', 'laccase', 'lactase']:
            csv_file = data_path / f"{dataset_name}.csv"
            
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    self._datasets[dataset_name] = df
                    logger.info(f"Loaded {dataset_name} dataset: {len(df)} samples")
                except Exception as e:
                    logger.error(f"Failed to load {dataset_name}: {e}")
            else:
                logger.warning(f"Dataset file not found: {csv_file}")
    
    def get_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Get a loaded dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame or None
        """
        return self._datasets.get(dataset_name)
    
    def is_loaded(self, dataset_name: str) -> bool:
        """Check if dataset is loaded.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            True if loaded
        """
        return dataset_name in self._datasets


# Global data service instance
data_service = DataService()
