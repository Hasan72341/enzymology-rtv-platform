import pandas as pd
import numpy as np
import joblib
import logging
import torch
import os
from typing import Dict, List, Any, Union
from pathlib import Path

# Important: Must import HybridEnsemble for joblib to load the model correctly
from src.models.hybrid_nn import HybridEnsemble
from src.features.esm_embeddings import ESM2Embedder
from src.features.scalar_features import ScalarFeatureEngineer

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self, config: Dict, model_name: str):
        """
        Initialize the Inference Engine.
        
        Args:
            config: Pipeline configuration dictionary
            model_name: The dataset/model to load (e.g., 'gst', 'laccase')
        """
        self.config = config
        self.model_name = model_name
        self.model_path = Path(f"outputs/models/{model_name}_selection_model.pkl")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}. You must run training first.")
        
        # Load the model and scaler
        data = joblib.load(self.model_path)
        if isinstance(data, dict) and 'model' in data:
            self.model = data['model']
            self.scaler = data.get('scaler')
        else:
            self.model = data
            self.scaler = None
            
        logger.info(f"Loaded model from {self.model_path}")
        
        # Initialize feature engineering components
        self.esm_embedder = ESM2Embedder(config['feature_engineering'])
        self.scalar_engineer = ScalarFeatureEngineer(config['feature_engineering'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                   'mps' if torch.backends.mps.is_available() else 'cpu')
                                   
    def predict_single(self, sequence: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Predict log_kcat for a single sequence."""
        df = pd.DataFrame([{'sequence': sequence}])
        
        if metadata:
            for k, v in metadata.items():
                df[k] = v
                
        # Fill missing required columns for scalar engineer with safe defaults
        defaults = {
            'ec_number': 'unknown',
            'organism': 'unknown',
            'molecularWeight': 0.0,
            'kmValue': 1.0,
            'ph_opt': 7.0,
            'temp_opt': 37.0,
            'n_measurements': 1
        }
        for k, v in defaults.items():
            if k not in df.columns:
                df[k] = v
                
        results = self._process_and_predict(df)
        return results.iloc[0].to_dict()
        
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict log_kcat for a batch of sequences in a DataFrame."""
        if 'sequence' not in df.columns:
            # Check lowercase
            seq_cols = [c for c in df.columns if c.lower() == 'sequence']
            if not seq_cols:
                raise ValueError("Input CSV must contain a 'sequence' column")
            df = df.rename(columns={seq_cols[0]: 'sequence'})
            
        # Drop rows with invalid sequences (too short or missing)
        original_len = len(df)
        df['sequence'] = df['sequence'].astype(str).str.strip().str.upper()
        df = df[df['sequence'].str.len() >= 10].copy()
        
        if len(df) < original_len:
            logger.warning(f"Dropped {original_len - len(df)} sequences that were too short or missing.")
            
        # Fill defaults for missing but expected columns
        defaults = {
            'ec_number': 'unknown',
            'organism': 'unknown',
            'molecularWeight': df['molecularWeight'].median() if 'molecularWeight' in df.columns else 0.0,
            'kmValue': df['kmValue'].median() if 'kmValue' in df.columns else 1.0,
            'ph_opt': df['ph_opt'].median() if 'ph_opt' in df.columns else 7.0,
            'temp_opt': df['temp_opt'].median() if 'temp_opt' in df.columns else 37.0,
            'n_measurements': 1
        }
        for k, v in defaults.items():
            if k not in df.columns:
                df[k] = v
                
        return self._process_and_predict(df)
        
    def _process_and_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal method to generate features and run prediction."""
        logger.info(f"Generating Scalar Features for {len(df)} sequences...")
        process_df = self.scalar_engineer.create_features(df)
        
        logger.info(f"Generating ESM-2 Embeddings ...")
        # Ensure model is on correct device
        if hasattr(self.esm_embedder, 'model') and self.esm_embedder.model is not None:
             self.esm_embedder.model.to(self.device)
             
        embeddings = self.esm_embedder.embed_sequences(process_df, self.model_name)
        
        intrinsic_features = self.scalar_engineer.get_intrinsic_features(process_df)
        X_intrinsic = process_df[intrinsic_features].values
        
        # Combine features
        X = np.hstack([embeddings, X_intrinsic])
        
        # Apply scaling if present
        if self.scaler is not None:
            X = self.scaler.transform(X)
            
        # Predict
        logger.info(f"Running predictions using model '{self.model_name}'...")
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(X)
        else:
            raise AttributeError("Loaded model does not have a predict() method.")
            
        # Add to df
        result_df = df.copy()
        result_df['predicted_log_kcat'] = predictions
        
        # Sort by predicted performance
        result_df = result_df.sort_values('predicted_log_kcat', ascending=False).reset_index(drop=True)
        return result_df
