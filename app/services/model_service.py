"""Model management service."""
import joblib
from pathlib import Path
from typing import Dict, Optional
import sys
import numpy as np
import pandas as pd

from app.config import settings
from app.utils import get_logger

# Add src to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.enzyme_selection import EnzymeSelectionModel

logger = get_logger(__name__)


class ModelService:
    """Service for loading and managing ML models."""
    
    _instance = None
    _models: Dict[str, EnzymeSelectionModel] = {}
    _model_info: Dict[str, dict] = {}
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_models(self) -> None:
        """Load all trained models from disk."""
        logger.info(f"Loading models from {settings.models_dir}")
        
        # Load agent.yaml to get model config
        import yaml
        try:
            with open(settings.config_file, 'r') as f:
                agent_config = yaml.safe_load(f)
            selection_config = agent_config['models']['enzyme_selection']
        except Exception as e:
            logger.error(f"Failed to load agent.yaml for model config: {e}")
            selection_config = {} # Fallback
            
        for model_name in settings.available_models:
            model_path = settings.models_dir / f"{model_name}_selection_model.pkl"
            
            try:
                if not model_path.exists():
                    logger.warning(f"Model file not found: {model_path}")
                    continue
                
                logger.info(f"Loading model: {model_name}")
                
                # Correct way: instantiate the class, then load the weights
                model_instance = EnzymeSelectionModel(selection_config)
                model_instance.load(str(model_path))
                
                self._models[model_name] = model_instance
                self._model_info[model_name] = {
                    "loaded": True,
                    "path": str(model_path),
                    "available": True
                }
                
                logger.info(f"Successfully loaded model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                self._model_info[model_name] = {
                    "loaded": False,
                    "error": str(e),
                    "available": False
                }
        
        logger.info(f"Loaded {len(self._models)} models")
    
    def get_model(self, dataset_name: str) -> Optional[EnzymeSelectionModel]:
        """Get a specific model.
        
        Args:
            dataset_name: Name of the dataset/model
            
        Returns:
            EnzymeSelectionModel instance or None
        """
        return self._models.get(dataset_name)
    
    def is_model_available(self, dataset_name: str) -> bool:
        """Check if a model is available.
        
        Args:
            dataset_name: Name of the dataset/model
            
        Returns:
            True if model is loaded and ready
        """
        return dataset_name in self._models
    
    def get_model_info(self, dataset_name: str) -> Optional[dict]:
        """Get model information.
        
        Args:
            dataset_name: Name of the dataset/model
            
        Returns:
            Model info dict or None
        """
        return self._model_info.get(dataset_name)
    
    def get_all_models_info(self) -> Dict[str, dict]:
        """Get information about all models.
        
        Returns:
            Dictionary of model info
        """
        return self._model_info.copy()
    
    def predict(self, model: EnzymeSelectionModel, features: np.ndarray) -> np.ndarray:
        """Generate predictions using a model.
        
        Args:
            model: Enzyme selection model
            features: Feature matrix
            
        Returns:
            Array of predictions
        """
        try:
            predictions = model.predict(features)
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def rank_enzymes(
        self,
        model: EnzymeSelectionModel,
        df: pd.DataFrame,
        predictions: np.ndarray,
        top_k: int = 10
    ) -> pd.DataFrame:
        """Rank enzymes by predicted activity.
        
        Args:
            model: Enzyme selection model
            df: DataFrame with enzyme data
            predictions: Predicted log_kcat values
            top_k: Number of top enzymes to return
            
        Returns:
            DataFrame of ranked enzymes
        """
        try:
            result_df = model.rank_enzymes(df, predictions, top_k=top_k)
            return result_df
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            raise


# Global model service instance
model_service = ModelService()
