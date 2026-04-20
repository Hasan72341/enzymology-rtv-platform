"""Feature engineering service."""
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from app.config import settings
from app.utils import get_logger
from app.schemas.requests import EnzymeData

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.features.esm_embeddings import ESM2Embedder
from src.features.scalar_features import ScalarFeatureEngineer

logger = get_logger(__name__)


class FeatureService:
    """Service for feature engineering."""
    
    def __init__(self):
        """Initialize feature service."""
        # Load config
        with open(settings.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize embedder and feature engineer
        self.embedder = ESM2Embedder(
            self.config['feature_engineering'],
            cache_dir=str(settings.embeddings_cache_dir)
        )
        self.feature_engineer = ScalarFeatureEngineer(
            self.config['feature_engineering']
        )
        
        logger.info("Feature service initialized")
    
    def enzyme_data_to_dataframe(self, enzyme_data: EnzymeData) -> pd.DataFrame:
        """Convert EnzymeData to DataFrame.
        
        Args:
            enzyme_data: Pydantic enzyme data model
            
        Returns:
            DataFrame with single row
        """
        data = {
            'sequence': [enzyme_data.sequence],
            'ec': [enzyme_data.ec if enzyme_data.ec else "0.0.0.0"],
            'organism': [enzyme_data.organism if enzyme_data.organism else "Unknown"],
            'n_measurements': [enzyme_data.n_measurements],
            'kcat_std': [enzyme_data.kcat_std],
            'kmValue': [enzyme_data.kmValue if enzyme_data.kmValue is not None else 1.0],
            'ph_opt': [enzyme_data.ph_opt if enzyme_data.ph_opt is not None else 7.0],
            'temp_opt': [enzyme_data.temp_opt if enzyme_data.temp_opt is not None else 30.0],
            'molecularWeight': [enzyme_data.molecularWeight if enzyme_data.molecularWeight is not None else 50000.0]
        }
        
        return pd.DataFrame(data)
    
    def enzymes_list_to_dataframe(self, enzymes: List[EnzymeData]) -> pd.DataFrame:
        """Convert list of EnzymeData to DataFrame.
        
        Args:
            enzymes: List of enzyme data
            
        Returns:
            DataFrame with multiple rows
        """
        dfs = [self.enzyme_data_to_dataframe(enzyme) for enzyme in enzymes]
        return pd.concat(dfs, ignore_index=True)
    
    def generate_features(
        self,
        df: pd.DataFrame,
        dataset_name: str
    ) -> tuple[np.ndarray, List[str]]:
        """Generate complete feature matrix.
        
        Args:
            df: DataFrame with enzyme data
            dataset_name: Dataset name for caching
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        logger.info(f"Generating features for {len(df)} enzymes")
        
        # Add scalar features
        df = self.feature_engineer.create_features(df)
        
        # Generate ESM-2 embeddings
        logger.info("Generating ESM-2 embeddings...")
        embeddings = self.embedder.embed_sequences(df, dataset_name)
        
        # Get intrinsic scalar features
        intrinsic_features = self.feature_engineer.get_intrinsic_features(df)
        X_intrinsic = df[intrinsic_features].values
        
        # Combine features
        X = np.hstack([embeddings, X_intrinsic])
        
        # Feature names
        feature_names = (
            [f'esm_dim_{i}' for i in range(embeddings.shape[1])] +
            intrinsic_features
        )
        
        logger.info(f"Generated {X.shape[1]} features ({embeddings.shape[1]} ESM + {len(intrinsic_features)} scalar)")
        
        return X, feature_names
    
    def get_process_features(self, df: pd.DataFrame) -> List[str]:
        """Get process feature names.
        
        Args:
            df: DataFrame with enzyme data
            
        Returns:
            List of process feature column names
        """
        return self.feature_engineer.get_process_features(df)


# Global feature service instance
feature_service = FeatureService()
