"""
Data preprocessing pipeline.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class EnzymePreprocessor:
    """Preprocess enzyme datasets according to configuration."""
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary from agent.yaml
        """
        self.config = config
        self.preproc_config = config['preprocessing']
        
    def preprocess(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline.
        
        Args:
            df: Raw enzyme DataFrame
            dataset_name: Name of the dataset for logging
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Starting preprocessing for {dataset_name}")
        logger.info(f"Initial shape: {df.shape}")
        
        # 1. Deduplication
        df = self._deduplicate(df)
        
        # 2. Filtering
        df = self._filter_sequences(df)
        df = self._filter_ec_samples(df)
        
        # 3. Numeric casting
        df = self._cast_numeric_columns(df)
        
        # 4. Missing value handling
        df = self._handle_missing_values(df)
        
        logger.info(f"Final shape after preprocessing: {df.shape}")
        return df
    
    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate duplicate records by UniProt ID.
        
        Instead of dropping duplicates, we aggregate measurements:
        - log_kcat: mean (combine measurements)
        - n_measurements: sum (total measurements across studies)
        - sequence: first (should be identical for same UniProt)
        - other numeric: mean
        - other string: first
        """
        # Check if deduplication is enabled
        if not self.preproc_config['deduplicate'].get('enabled', True):
            logger.info("Deduplication disabled, keeping all records")
            return df
        
        dedup_col = self.preproc_config['deduplicate']['by']
        initial_len = len(df)
        
        # Group by UniProt ID
        grouped = df.groupby(dedup_col, as_index=False)
        
        # Define aggregation strategy
        agg_dict = {}
        for col in df.columns:
            if col == dedup_col:
                continue  # Skip grouping column
            elif col == 'log_kcat':
                agg_dict[col] = 'mean'  # Average activity across measurements
            elif col == 'n_measurements':
                agg_dict[col] = 'sum'  # Total measurements
            elif col == 'kcat_std':
                agg_dict[col] = 'mean'  # Average std
            elif col in ['sequence', 'organism', 'ec', 'dataset_name', 'ec_number', 'real_world_problem']:
                agg_dict[col] = 'first'  # Categorical - take first
            elif df[col].dtype in ['float64', 'int64']:
                agg_dict[col] = 'mean'  # Numeric - average
            else:
                agg_dict[col] = 'first'  # Default - take first
        
        df_dedup = grouped.agg(agg_dict)
        
        removed = initial_len - len(df_dedup)
        if removed > 0:
            logger.info(
                f"Aggregated {initial_len} records into {len(df_dedup)} unique enzymes "
                f"(by '{dedup_col}', averaged log_kcat across measurements)"
            )
        
        return df_dedup
    
    def _filter_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter sequences by minimum length."""
        min_length = self.preproc_config['filter']['min_sequence_length']
        initial_len = len(df)
        
        df = df.copy()  # Avoid SettingWithCopyWarning
        df['sequence_length'] = df['sequence'].str.len()
        df = df[df['sequence_length'] >= min_length].copy()
        
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} sequences shorter than {min_length} AA")
        
        return df
    
    def _filter_ec_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter EC numbers with insufficient samples."""
        min_samples = self.preproc_config['filter']['min_samples_per_ec']
        initial_len = len(df)
        
        # Count samples per EC
        ec_counts = df['ec'].value_counts()
        valid_ecs = ec_counts[ec_counts >= min_samples].index
        
        df = df[df['ec'].isin(valid_ecs)].copy()
        
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} records from ECs with < {min_samples} samples")
        
        return df
    
    def _cast_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast specified columns to numeric, coercing errors to NaN."""
        numeric_cols = self.preproc_config['numeric_cast']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.debug(f"Cast column '{col}' to numeric")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values: drop or impute based on strategy."""
        strategy = self.preproc_config['missing_value_strategy']
        
        # Drop rows with missing critical columns
        drop_cols = strategy['drop_if_missing']
        initial_len = len(df)
        df = df.dropna(subset=drop_cols)
        
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Dropped {removed} rows with missing {drop_cols}")
        
        # Impute missing optional columns
        impute_strategy = strategy['impute_if_missing']
        for col, method in impute_strategy.items():
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    if method == 'median':
                        impute_value = df[col].median()
                        df[col] = df[col].fillna(impute_value)
                        logger.info(
                            f"Imputed {missing_count} missing '{col}' values with median: {impute_value:.2f}"
                        )
        
        return df
