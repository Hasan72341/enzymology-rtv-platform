"""
Model B: Bioprocess Optimization
Optimizes pH and temperature for maximum predicted activity.
"""
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class BioprocessOptimizer:
    """Model B: Optimize pH and temperature conditions."""
    
    def __init__(self, config: Dict):
        """
        Initialize bioprocess optimizer.
        
        Args:
            config: Model configuration from agent.yaml
        """
        self.config = config
        self.target = config['target']  # 'log_kcat'
        self.model_list = config['models']
        
        # Optimization grid
        opt_config = config.get('optimization', {})
        self.ph_steps = opt_config.get('ph_steps', 50)
        self.temp_steps = opt_config.get('temperature_steps', 50)
        self.stability_lambda = opt_config.get('stability_penalty', {}).get('lambda', 0.1)
        
        # Constraints
        constraints = config.get('constraints', {})
        self.ph_margin = constraints.get('ph_range', {}).get('margin', 1.5)
        self.temp_margin = constraints.get('temperature_range', {}).get('margin', 10)
        
        self.model = None
        
    def train(
        self, 
        X: np.ndarray,  # [ph, temperature]
        y: np.ndarray,  # log_kcat
        model_type: str = 'gaussian_process'
    ) -> Dict:
        """
        Train optimization model.
        
        Args:
            X: Process conditions (pH, temperature)
            y: Target activity (log_kcat)
            model_type: 'gaussian_process' or 'xgboost'
            
        Returns:
            Training metrics
        """
        logger.info(f"Training bioprocess optimizer with {model_type}...")
        
        # Scale features to prevent kernel overflow/NaNs
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        if model_type == 'gaussian_process':
            # Gaussian Process with RBF kernel and normalized target
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.1,
                normalize_y=True,
                n_restarts_optimizer=10,
                random_state=42
            )
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X, y)
        
        # Evaluate
        y_pred = self.model.predict(X)
        from sklearn.metrics import r2_score, mean_squared_error
        
        metrics = {
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        logger.info(f"Model performance - R²: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.3f}")
        
        return metrics
    
    def optimize(
        self,
        df: pd.DataFrame,
        ph_opt: float = None,
        temp_opt: float = None
    ) -> Tuple[float, float, float]:
        """
        Find optimal pH and temperature via grid search.
        
        Args:
            df: DataFrame with pH and temperature data
            ph_opt: Known pH optimum (for constraints)
            temp_opt: Known temperature optimum (for constraints)
            
        Returns:
            (optimal_ph, optimal_temp, predicted_log_kcat)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Determine search bounds
        if ph_opt is not None:
            ph_range = (
                max(0, ph_opt - self.ph_margin),
                min(14, ph_opt + self.ph_margin)
            )
        else:
            ph_range = (df['ph_opt'].min(), df['ph_opt'].max()) if 'ph_opt' in df.columns else (3, 8)
        
        if temp_opt is not None:
            temp_range = (
                max(0, temp_opt - self.temp_margin),
                min(100, temp_opt + self.temp_margin)
            )
        else:
            temp_range = (df['temp_opt'].min(), df['temp_opt'].max()) if 'temp_opt' in df.columns else (25, 70)
        
        logger.info(f"Optimizing over pH: {ph_range}, Temperature: {temp_range}")
        
        # Create grid
        ph_grid = np.linspace(ph_range[0], ph_range[1], self.ph_steps)
        temp_grid = np.linspace(temp_range[0], temp_range[1], self.temp_steps)
        
        PH, TEMP = np.meshgrid(ph_grid, temp_grid)
        conditions = np.column_stack([PH.ravel(), TEMP.ravel()])
        
        # Predict on grid
        scaled_conditions = self.scaler.transform(conditions) if hasattr(self, 'scaler') else conditions
        predictions = self.model.predict(scaled_conditions)
        
        # Apply stability penalty (penalize distance from known optima)
        if ph_opt is not None and temp_opt is not None:
            distances = np.sqrt(
                ((conditions[:, 0] - ph_opt) / self.ph_margin) ** 2 +
                ((conditions[:, 1] - temp_opt) / self.temp_margin) ** 2
            )
            penalized_predictions = predictions - self.stability_lambda * distances
        else:
            penalized_predictions = predictions
        
        # Find optimum
        best_idx = np.argmax(penalized_predictions)
        optimal_ph = conditions[best_idx, 0]
        optimal_temp = conditions[best_idx, 1]
        predicted_activity = predictions[best_idx]
        
        logger.info(
            f"Optimal conditions: pH={optimal_ph:.2f}, T={optimal_temp:.1f}°C, "
            f"Predicted log_kcat={predicted_activity:.3f}"
        )
        
        return optimal_ph, optimal_temp, predicted_activity
    
    def predict_heatmap(
        self,
        ph_range: Tuple[float, float],
        temp_range: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate prediction heatmap for visualization.
        
        Args:
            ph_range: (min_ph, max_ph)
            temp_range: (min_temp, max_temp)
            
        Returns:
            (pH grid, temperature grid, predictions)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        ph_grid = np.linspace(ph_range[0], ph_range[1], self.ph_steps)
        temp_grid = np.linspace(temp_range[0], temp_range[1], self.temp_steps)
        
        PH, TEMP = np.meshgrid(ph_grid, temp_grid)
        conditions = np.column_stack([PH.ravel(), TEMP.ravel()])
        
        scaled_conditions = self.scaler.transform(conditions) if hasattr(self, 'scaler') else conditions
        predictions = self.model.predict(scaled_conditions)
        predictions_2d = predictions.reshape(PH.shape)
        
        return PH, TEMP, predictions_2d
