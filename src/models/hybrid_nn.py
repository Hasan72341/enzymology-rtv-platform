import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import xgboost as xgb

logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    """A standard residual block for deep feed-forward networks."""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        
    def forward(self, x):
        # Identity skip connection
        return nn.functional.relu(x + self.block(x))


class GaussianNoise(nn.Module):
    """Injects small random noise to prevent overfitting on exact numerical embeddings."""
    def __init__(self, std_dev: float = 0.02):
        super().__init__()
        self.std_dev = std_dev

    def forward(self, x):
        if self.training and self.std_dev > 0:
            return x + torch.randn_like(x) * self.std_dev
        return x


class ResidualFFN(nn.Module):
    """PyTorch FFN with residual connections and Batch Normalization."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_blocks: int = 2):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            GaussianNoise(std_dev=0.02),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        out = self.input_layer(x)
        
        for block in self.res_blocks:
            residual = out
            out = block(out)
            out = F.gelu(out + residual)
            
        out = self.output_layer(out)
        return out.squeeze(-1)


class HybridEnsemble:
    """An innovative hybrid model combining a PyTorch Residual FFN and classical XGBoost."""
    def __init__(self, input_dim: int, n_epochs: int = 40, lr: float = 1e-3, 
                 hidden_dim: int = 256, num_blocks: int = 2,
                 xgb_n_estimators: int = 150, xgb_max_depth: int = 6, xgb_lr: float = 0.05,
                 xgb_weight: float = 0.6):
        self.input_dim = input_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_lr = xgb_lr
        self.xgb_weight = xgb_weight
        
        self.ffn = ResidualFFN(input_dim, hidden_dim=hidden_dim, num_blocks=num_blocks)
        self.xgb = xgb.XGBRegressor(
            n_estimators=xgb_n_estimators, 
            max_depth=xgb_max_depth, 
            learning_rate=xgb_lr, 
            random_state=42
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ffn.to(self.device)
        self.trained = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        logger.info("Training the HybridEnsemble (XGBoost + FFN)...")
        # 1. Train Classical XGBoost
        self.xgb.fit(X, y, sample_weight=sample_weight)
        
        # 2. Train FFN mapping
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        if sample_weight is not None:
            w_t = torch.tensor(sample_weight, dtype=torch.float32).to(self.device)
            dataset = TensorDataset(X_t, y_t, w_t)
        else:
            w_t = torch.ones_like(y_t)
            dataset = TensorDataset(X_t, y_t, w_t)
            
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        optimizer = optim.Adam(self.ffn.parameters(), lr=self.lr)
        criterion = nn.MSELoss(reduction='none')

        for epoch in range(self.n_epochs):
            self.ffn.train()
            for batch_X, batch_y, batch_w in loader:
                # Handle single sample inputs to BatchNorm robustly
                if batch_X.size(0) <= 1:
                    continue
                    
                # Optional Mixup Data Augmentation inside batch
                if np.random.rand() < 0.5: # 50% chance to mixup
                    alpha = 0.2
                    lam = np.random.beta(alpha, alpha)
                    batch_size = batch_X.size(0)
                    index = torch.randperm(batch_size).to(self.device)
                    
                    batch_X = lam * batch_X + (1 - lam) * batch_X[index]
                    batch_y = lam * batch_y + (1 - lam) * batch_y[index]
                    batch_w = lam * batch_w + (1 - lam) * batch_w[index]
                    
                optimizer.zero_grad()
                preds = self.ffn(batch_X)
                loss = criterion(preds, batch_y)
                loss = (loss * batch_w).mean()  # Incorporate sample weights
                loss.backward()
                optimizer.step()
                
        self.trained = True
        return self

    def predict(self, X: np.ndarray):
        if not self.trained:
            raise ValueError("HybridEnsemble is not trained yet.")
        # Classical prediction
        pred_xgb = self.xgb.predict(X)
        
        # Neural prediction
        self.ffn.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred_ffn = self.ffn(X_t).cpu().numpy()
            
        # Ensemble arithmetic mean gives highly robust generalization
        return self.xgb_weight * pred_xgb + (1 - self.xgb_weight) * pred_ffn
        
    def get_params(self, deep=True):
        """Scikit-learn compat shim."""
        return {
            "input_dim": self.input_dim,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "xgb_n_estimators": self.xgb_n_estimators,
            "xgb_max_depth": self.xgb_max_depth,
            "xgb_lr": self.xgb_lr,
            "xgb_weight": self.xgb_weight
        }
        
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
