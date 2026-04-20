import optuna
import numpy as np
import logging
import torch
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from .hybrid_nn import HybridEnsemble, ResidualFFN
import xgboost as xgb

logger = logging.getLogger(__name__)

class HybridHyperparameterOptimizer:
    """Uses Optuna to optimize hyperparameters for the Hybrid ML Ensembles."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray = None, n_trials: int = 30):
        self.X = X
        self.y = y
        self.sample_weights = sample_weights
        self.n_trials = n_trials
        
    def objective(self, trial) -> float:
        # Suggest FFN params
        ffn_hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        ffn_num_blocks = trial.suggest_int("num_blocks", 1, 4)
        ffn_lr = trial.suggest_float("ffn_lr", 1e-4, 1e-2, log=True)
        
        # Suggest XGBoost params
        xgb_lr = trial.suggest_float("xgb_lr", 1e-3, 0.3, log=True)
        xgb_depth = trial.suggest_int("xgb_max_depth", 3, 10)
        xgb_estimators = trial.suggest_int("xgb_n_estimators", 50, 300, step=50)
        
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in cv.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            w_train = self.sample_weights[train_idx] if self.sample_weights is not None else None
            
            # Instantiate model
            model = HybridEnsemble(input_dim=self.X.shape[1], n_epochs=30, lr=ffn_lr)
            
            # Patch sub-models securely with trial parameters
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.ffn = ResidualFFN(self.X.shape[1], hidden_dim=ffn_hidden_dim, num_blocks=ffn_num_blocks).to(device)
            model.xgb = xgb.XGBRegressor(
                n_estimators=xgb_estimators, 
                max_depth=xgb_depth, 
                learning_rate=xgb_lr, 
                random_state=42
            )
            
            # Train and predict
            model.fit(X_train, y_train, sample_weight=w_train)
            preds = model.predict(X_val)
            
            # Evaluate (Spearman represents relative ranking effectiveness)
            spearman, _ = spearmanr(y_val, preds)
            # Handle NaN correlation if constant preds
            if np.isnan(spearman):
                spearman = -1.0
            scores.append(spearman)
            
            # Optionally prune bad trials early
            trial.report(np.mean(scores), len(scores))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        return float(np.mean(scores))
        
    def optimize(self) -> dict:
        """Run the optimization study to find maximum rank alignment."""
        logger.info(f"Starting Optuna search over {self.n_trials} trials for HybridEnsemble...")
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(self.objective, n_trials=self.n_trials)
        
        logger.info("================ OPTIMIZATION FINISHED ================")
        logger.info(f"Best Spearman correlation: {study.best_value:.4f}")
        logger.info(f"Best Configuration: {study.best_params}")
        return study.best_params

if __name__ == "__main__":
    # Provides testability block natively
    print("Initialize HybridHyperparameterOptimizer instance from PipelineOrchestrator to run.")
