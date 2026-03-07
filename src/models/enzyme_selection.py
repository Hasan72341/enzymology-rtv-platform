"""
Model A: Enzyme Selection
Predicts log_kcat using sequence embeddings and metadata.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import xgboost as xgb
from typing import Dict, Tuple, List
import logging
import joblib

logger = logging.getLogger(__name__)


class EnzymeSelectionModel:
    """Model A: Select best enzyme variants based on predicted log_kcat."""
    
    def __init__(self, config: Dict):
        """
        Initialize enzyme selection model.
        
        Args:
            config: Model configuration from agent.yaml
        """
        self.config = config
        self.model_configs = config['models']
        self.target = config['target']  # 'log_kcat'
        self.sample_weight_col = config.get('sample_weight', 'n_measurements')
        self.top_k = config['output']['top_k']
        
        self.models = {}
        self.best_model_name = None
        self.best_model = None
        
    def train(
        self, 
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        sample_weights: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict:
        """
        Train all models with EC-aware cross-validation.
        
        Args:
            X: Feature matrix (embeddings + scalar features)
            y: Target values (log_kcat)
            groups: Group labels for GroupKFold (EC numbers)
            sample_weights: Sample weights (n_measurements)
            feature_names: Names of features for interpretability
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Training models on {len(X)} samples...")
        
        # Normalize features to prevent numerical overflow (especially Ridge/GP)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        results = {}
        
        # Initialize models
        model_list = self.config['models']
        
        for model_name in model_list:
            logger.info(f"Training {model_name}...")
            
            if model_name == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_name == 'xgboost':
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_name == 'linear_probe':
                model = Ridge(alpha=100.0, solver='svd', random_state=42)
            elif model_name == 'hybrid_ensemble':
                from src.models.hybrid_nn import HybridEnsemble, ResidualFFN
                from src.models.hyperparameter_search import HybridHyperparameterOptimizer
                import torch
                
                logger.info("Automated Optuna Hyperparameter Tuning initialized for Hybrid Ensemble...")
                optimizer = HybridHyperparameterOptimizer(X, y, sample_weights, n_trials=10)
                best_params = optimizer.optimize()
                
                model = HybridEnsemble(
                    input_dim=X.shape[1], 
                    n_epochs=30, 
                    lr=best_params['ffn_lr'],
                    hidden_dim=best_params['hidden_dim'],
                    num_blocks=best_params['num_blocks'],
                    xgb_n_estimators=best_params['xgb_n_estimators'],
                    xgb_max_depth=best_params['xgb_max_depth'],
                    xgb_lr=best_params['xgb_lr']
                )
            else:
                logger.warning(f"Unknown model: {model_name}, skipping")
                continue
            
            # Cross-validation with EC-aware splits
            cv_scores = self._cross_validate(
                model, X, y, groups, sample_weights, n_splits=5
            )
            
            # Train final model on all data
            logger.info(f"Synthesizing tabular data (SMOTE) to boost robustness for {model_name}...")
            X_aug, y_aug, w_aug = self.synthesize_tabular_data(X, y, sample_weights, multiplier=1)
            model.fit(X_aug, y_aug, sample_weight=w_aug)
            
            # Store model and results
            self.models[model_name] = model
            results[model_name] = cv_scores
            
            logger.info(
                f"{model_name} - R²: {cv_scores['r2']:.3f}, "
                f"RMSE: {cv_scores['rmse']:.3f}, "
                f"Spearman: {cv_scores['spearman']:.3f}"
            )
        
        # Select best model based on Spearman correlation
        best_name = max(results.keys(), key=lambda k: results[k]['spearman'])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        logger.info(f"Best model: {best_name} (Spearman: {results[best_name]['spearman']:.3f})")
        
        return results
    
    def synthesize_tabular_data(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, multiplier: int = 1) -> tuple:
        """Applies a robust SMOTE-like continuous interpolation to augment the dataset size."""
        n_samples = len(X)
        if n_samples < 2 or multiplier <= 0:
            return X, y, weights
            
        n_synthetic = n_samples * multiplier
        syn_x, syn_y, syn_w = np.zeros((n_synthetic, X.shape[1])), np.zeros(n_synthetic), np.zeros(n_synthetic)
        
        for i in range(n_synthetic):
            idx1, idx2 = np.random.randint(0, n_samples, size=2)
            alpha = np.random.random()
            syn_x[i] = X[idx1] + alpha * (X[idx2] - X[idx1])
            syn_y[i] = y[idx1] + alpha * (y[idx2] - y[idx1])
            syn_w[i] = weights[idx1] + alpha * (weights[idx2] - weights[idx1])
            
        return np.vstack([X, syn_x]), np.concatenate([y, syn_y]), np.concatenate([weights, syn_w])

    def _cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        sample_weights: np.ndarray,
        n_splits: int = 5
    ) -> Dict:
        """
        Perform EC-aware cross-validation.
        
        Args:
            model: Sklearn-compatible model
            X, y, groups, sample_weights: Training data
            n_splits: Number of CV folds
            
        Returns:
            Dictionary of averaged metrics
        """
        from sklearn.metrics import mean_absolute_error, median_absolute_error

        # Check if we have enough groups for GroupKFold
        n_groups = len(np.unique(groups))
        n_samples = len(X)
        
        # Skip CV for very small datasets
        if n_samples < 10:
            logger.warning(
                f"Dataset too small ({n_samples} samples) for cross-validation. "
                f"Training on all data without CV."
            )
            # Train on all data
            model.fit(X, y, sample_weight=sample_weights)
            y_pred = model.predict(X)
            
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            med_ae = median_absolute_error(y, y_pred)
            spearman, _ = spearmanr(y, y_pred)
            spearman = np.nan_to_num(spearman, nan=0.0)
            
            return {
                'r2': r2, 'r2_std': 0.0, 'r2_per_fold': [r2],
                'rmse': rmse, 'rmse_std': 0.0, 'rmse_per_fold': [rmse],
                'mse': mse, 'mse_std': 0.0, 'mse_per_fold': [mse],
                'mae': mae, 'mae_std': 0.0, 'mae_per_fold': [mae],
                'spearman': spearman, 'spearman_std': 0.0, 'spearman_per_fold': [spearman],
                'median_ae': med_ae, 'median_ae_std': 0.0, 'median_ae_per_fold': [med_ae]
            }
        
        if n_groups < n_splits:
            actual_splits = min(max(n_groups, 2), 3)
            logger.warning(
                f"Only {n_groups} unique EC groups, using regular KFold({actual_splits}) instead of GroupKFold"
            )
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=min(max(n_groups, 2), 3), shuffle=True, random_state=42)
            splits = cv.split(X)
        else:
            from sklearn.model_selection import GroupKFold
            gkf = GroupKFold(n_splits=min(n_splits, n_groups))
            splits = gkf.split(X, y, groups)
        
        r2_scores, rmse_scores, mse_scores = [], [], []
        mae_scores, med_ae_scores, spearman_scores = [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            w_train = sample_weights[train_idx]
            
            # Clone and train model
            import copy
            from sklearn.base import clone
            try:
                fold_model = clone(model)
            except Exception:
                # Fallback for HybridEnsemble which might lack get_params
                fold_model = copy.deepcopy(model)
            
            # Apply tabular data augmentation strictly on the training fold
            X_t_aug, y_t_aug, w_t_aug = self.synthesize_tabular_data(X_train, y_train, w_train, multiplier=1)
            fold_model.fit(X_t_aug, y_t_aug, sample_weight=w_t_aug)
            
            # Predict
            y_pred = fold_model.predict(X_val)
            
            # Metrics
            r2 = r2_score(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val, y_pred)
            med_ae = median_absolute_error(y_val, y_pred)
            spearman, _ = spearmanr(y_val, y_pred)
            spearman = np.nan_to_num(spearman, nan=0.0)
            
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            mse_scores.append(mse)
            mae_scores.append(mae)
            med_ae_scores.append(med_ae)
            spearman_scores.append(spearman)
        
        return {
            'r2': np.mean(r2_scores), 'r2_std': np.std(r2_scores), 'r2_per_fold': r2_scores,
            'rmse': np.mean(rmse_scores), 'rmse_std': np.std(rmse_scores), 'rmse_per_fold': rmse_scores,
            'mse': np.mean(mse_scores), 'mse_std': np.std(mse_scores), 'mse_per_fold': mse_scores,
            'mae': np.mean(mae_scores), 'mae_std': np.std(mae_scores), 'mae_per_fold': mae_scores,
            'spearman': np.mean(spearman_scores), 'spearman_std': np.std(spearman_scores), 'spearman_per_fold': spearman_scores,
            'median_ae': np.mean(med_ae_scores), 'median_ae_std': np.std(med_ae_scores), 'median_ae_per_fold': med_ae_scores
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict log_kcat using best model."""
        if self.best_model is None:
            raise ValueError("Model not trained yet")
        if hasattr(self, 'scaler') and self.scaler is not None:
            X = self.scaler.transform(X)
        return self.best_model.predict(X)
    
    def rank_enzymes(self, df: pd.DataFrame, predictions: np.ndarray, top_k: int = None) -> pd.DataFrame:
        """
        Rank enzymes by predicted log_kcat.
        
        Args:
            df: Original DataFrame with metadata
            predictions: Predicted log_kcat values
            top_k: Number of top enzymes to return (defaults to config value if None)
            
        Returns:
            DataFrame of top-k ranked enzymes
        """
        results = df.copy()
        results['predicted_log_kcat'] = predictions
        results['rank'] = results['predicted_log_kcat'].rank(ascending=False)
        
        # Get top k
        k = top_k if top_k is not None else self.top_k
        top_enzymes = results.nsmallest(k, 'rank')
        
        cols = ['uniprot_primary', 'organism', 'predicted_log_kcat', 'rank', 'sequence']
        if 'log_kcat' in results.columns:
            cols.append('log_kcat')
            
        return top_enzymes[cols]
    
    def save(self, path: str):
        """Save best model and scaler."""
        if self.best_model is None:
            raise ValueError("No model to save")
        save_data = {
            'model': self.best_model,
            'scaler': getattr(self, 'scaler', None),
            'best_model_name': self.best_model_name,
        }
        joblib.dump(save_data, path)
        logger.info(f"Saved model to {path}")
    
    def load(self, path: str):
        """Load trained model (backward-compatible with old .pkl files)."""
        data = joblib.load(path)
        if isinstance(data, dict) and 'model' in data:
            self.best_model = data['model']
            self.scaler = data.get('scaler')
            self.best_model_name = data.get('best_model_name')
        else:
            # Backward compat: old .pkl files saved raw model only
            self.best_model = data
            self.scaler = None
        logger.info(f"Loaded model from {path}")
