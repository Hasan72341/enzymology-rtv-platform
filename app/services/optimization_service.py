"""Bioprocess optimization service."""
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from app.config import settings
from app.utils import get_logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.bioprocess_optimization import BioprocessOptimizer

logger = get_logger(__name__)


class OptimizationService:
    """Service for bioprocess optimization."""
    
    def __init__(self):
        """Initialize optimization service."""
        # Load config
        with open(settings.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.bioprocess_config = self.config['models']['bioprocess_optimization']
        
        logger.info("Optimization service initialized")
    
    def optimize_conditions(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        process_features: list,
        ph_range: Optional[Tuple[float, float]] = None,
        temp_range: Optional[Tuple[float, float]] = None
    ) -> Dict:
        """Optimize pH and temperature conditions.
        
        Args:
            df: DataFrame with enzyme data
            y: Target values (log_kcat)
            process_features: List of process feature names
            ph_range: Optional pH range
            temp_range: Optional temperature range
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting bioprocess optimization")
        
        # Check data sufficiency
        MIN_SAMPLES_FOR_OPTIMIZATION = 20
        n_samples = len(df)
        
        if len(process_features) < 2 or n_samples < MIN_SAMPLES_FOR_OPTIMIZATION:
            logger.warning(
                f"Insufficient data for optimization: {n_samples} samples, "
                f"{len(process_features)} process features"
            )
            return {
                'skipped': True,
                'reason': f'Insufficient data ({n_samples} samples, need ≥{MIN_SAMPLES_FOR_OPTIMIZATION})',
                'sufficient_variation': False,
                'n_samples': n_samples
            }
        
        # Prepare process features
        X_process = df[process_features[:2]].values  # [ph, temp]
        
        # Create optimizer
        optimizer = BioprocessOptimizer(self.bioprocess_config)
        
        # Train Gaussian Process model
        optimizer.train(X_process, y, model_type='gaussian_process')
        
        # Get baseline values
        ph_opt_baseline = df[process_features[0]].median()
        temp_opt_baseline = df[process_features[1]].median()
        
        # Optimize
        optimal_ph, optimal_temp, predicted_activity = optimizer.optimize(
            df, ph_opt_baseline, temp_opt_baseline
        )
        
        # Calculate variation
        ph_variation = df[process_features[0]].max() - df[process_features[0]].min()
        temp_variation = df[process_features[1]].max() - df[process_features[1]].min()
        
        sufficient_variation = ph_variation > 0.5 or temp_variation > 5
        
        # Calculate improvement against median baseline
        baseline_log_kcat = y.mean()
        improvement = None
        if baseline_log_kcat != 0:
            improvement = ((predicted_activity - baseline_log_kcat) / abs(baseline_log_kcat)) * 100
        
        return {
            'optimal_ph': float(optimal_ph),
            'optimal_temperature': float(optimal_temp),
            'predicted_log_kcat': float(predicted_activity),
            'baseline_log_kcat': float(baseline_log_kcat),
            'improvement_percentage': float(improvement) if improvement is not None else None,
            'ph_range': [float(df[process_features[0]].min()), float(df[process_features[0]].max())],
            'temp_range': [float(df[process_features[1]].min()), float(df[process_features[1]].max())],
            'sufficient_variation': sufficient_variation,
            'n_samples': n_samples,
            'skipped': False
        }
    
    def generate_heatmap(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        process_features: list,
        ph_range: Tuple[float, float],
        temp_range: Tuple[float, float]
    ) -> Optional[Dict]:
        """Generate pH-temperature heatmap data.
        
        Args:
            df: DataFrame with enzyme data
            y: Target values (log_kcat)
            process_features: List of process feature names
            ph_range: pH range
            temp_range: Temperature range
            
        Returns:
            Dictionary with heatmap data or None
        """
        if len(process_features) < 2:
            return None
        
        X_process = df[process_features[:2]].values
        
        # Create and train optimizer
        optimizer = BioprocessOptimizer(self.bioprocess_config)
        optimizer.train(X_process, y, model_type='gaussian_process')
        
        # Generate heatmap
        ph_grid, temp_grid, pred_heatmap = optimizer.predict_heatmap(ph_range, temp_range)
        
        return {
            'ph_grid': ph_grid.tolist(),
            'temp_grid': temp_grid.tolist(),
            'predictions': pred_heatmap.tolist()
        }


# Global optimization service instance
optimization_service = OptimizationService()
