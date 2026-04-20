"""
Scalar feature engineering for enzyme data.
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ScalarFeatureEngineer:
    """Create scalar features from enzyme metadata."""
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration from agent.yaml
        """
        self.config = config['scalar_features']
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all scalar features.
        
        Args:
            df: DataFrame with enzyme data
            
        Returns:
            DataFrame with additional feature columns
        """
        df = df.copy()
        
        # Intrinsic features
        df = self._create_log_kmvalue(df)
        df = self._create_ec_encoding(df)
        
        # Organism features (if available)
        df = self._create_organism_features(df)
        
        logger.info(f"Created scalar features. New shape: {df.shape}")
        return df
    
    def _create_log_kmvalue(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log-transform kmValue."""
        if 'kmValue' in df.columns:
            # Add small constant to avoid log(0)
            df['log_kmValue'] = np.log10(df['kmValue'] + 1e-6)
            logger.debug("Created log_kmValue feature")
        return df
    
    def _create_ec_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create EC number encoding features.
        
        EC numbers have format: X.X.X.X (4 levels)
        """
        # Always create all 4 EC level features
        for level in ['ec_level1', 'ec_level2', 'ec_level3', 'ec_level4']:
            df[level] = 0
        
        if 'ec' in df.columns:
            # Split EC into levels
            ec_split = df['ec'].str.split('.', expand=True)
            
            for i, level_name in enumerate(['ec_level1', 'ec_level2', 'ec_level3', 'ec_level4']):
                if i < ec_split.shape[1]:
                    df[level_name] = pd.to_numeric(ec_split[i], errors='coerce').fillna(0).astype(int)
        
        logger.debug("Created EC encoding features")
        return df
    
    def _create_organism_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create organism-based features.
        
        Simple approach: extract organism kingdom/domain indicators
        """
        # Always create organism features, default to 0
        df['is_bacteria'] = 0
        df['is_fungi'] = 0
        
        if 'organism' in df.columns:
            # Common organism categories (very simplified)
            organism_lower = df['organism'].str.lower().fillna('')
            
            # Bacteria indicators
            bacteria_keywords = ['bacillus', 'escherichia', 'streptomyces', 'bacterium']
            df['is_bacteria'] = organism_lower.apply(
                lambda x: 1 if any(kw in str(x) for kw in bacteria_keywords) else 0
            )
            
            # Fungi indicators
            fungi_keywords = ['trametes', 'pleurotus', 'coprin', 'laccase', 'myc']
            df['is_fungi'] = organism_lower.apply(
                lambda x: 1 if any(kw in str(x) for kw in fungi_keywords) else 0
            )
        
        logger.debug("Created organism features")
        return df
    
    def get_intrinsic_features(self, df: pd.DataFrame) -> list:
        """Get list of intrinsic feature column names."""
        features = []
        
        # Add configured intrinsic features if they exist
        for feat in self.config['intrinsic']:
            if feat == 'log_kmValue' and 'log_kmValue' in df.columns:
                features.append('log_kmValue')
            elif feat in df.columns:
                features.append(feat)
        
        # Always add EC features in consistent order
        for level in ['ec_level1', 'ec_level2', 'ec_level3', 'ec_level4']:
            if level in df.columns:
                features.append(level)
        
        # Always add organism features in consistent order
        for org_feat in ['is_bacteria', 'is_fungi']:
            if org_feat in df.columns:
                features.append(org_feat)
        
        return features
    
    def get_process_features(self, df: pd.DataFrame) -> list:
        """Get list of process-related feature column names."""
        features = []
        
        for feat in self.config['process']:
            # Map 'ph' to actual column names
            if feat == 'ph':
                if 'ph_opt' in df.columns:
                    features.append('ph_opt')
                elif 'ph_max' in df.columns:
                    features.append('ph_max')
            elif feat == 'temperature':
                if 'temp_opt' in df.columns:
                    features.append('temp_opt')
                elif 'temp_max' in df.columns:
                    features.append('temp_max')
            elif feat in df.columns:
                features.append(feat)
        
        return features
