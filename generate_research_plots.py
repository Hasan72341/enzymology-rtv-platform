import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add current dir to path
sys.path.insert(0, os.getcwd())

from src.models.bioprocess_optimization import BioprocessOptimizer

# --- APPLE GRADE STYLING ---
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#D2D2D7", # Apple Gray
    "axes.labelcolor": "#1D1D1F",
    "text.color": "#1D1D1F",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.color": "#F5F5F7",
    "grid.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

APPLE_BLUE = "#0071E3"
APPLE_GREEN = "#28CD41"
APPLE_RED = "#FF3B30"
APPLE_GRAY = "#F5F5F7"

def generate_research_plots():
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = ["gst", "laccase", "lactase"]
    optima = {"gst": (6.97, 29.8), "laccase": (4.72, 54.8), "lactase": (6.52, 48.8)}
    
    for ds in datasets:
        print(f"Generating Apple-grade visuals for {ds}...")
        results_path = Path(f"outputs/reports/{ds}_full_results.json")
        data_path = Path(f"data/{ds}.csv")
        if not results_path.exists(): continue
            
        with open(results_path, 'r') as f:
            res = json.load(f)

        # 1. Feature Importance (Minimalist Apple Style)
        fi = res.get('feature_importance', {})
        if fi:
            fi_series = pd.Series(fi).sort_values(ascending=False).head(15)
            plt.figure(figsize=(9, 7))
            # Use monochromatic blue gradient for a "Premium" feel
            colors = sns.light_palette(APPLE_BLUE, n_colors=len(fi_series), reverse=True)
            sns.barplot(x=fi_series.values, y=fi_series.index, palette=colors, edgecolor=None)
            plt.title(f"Predictive Feature Contribution: {ds.upper()}", fontsize=18, fontweight='bold', pad=25)
            plt.xlabel("Relative Information Gain", fontsize=12, fontweight='medium', color="#86868B")
            plt.ylabel("Latent Evolutionary Dimension", fontsize=12, fontweight='medium', color="#86868B")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{ds}_feature_importance.png")
            plt.close()

        # 2. Optimization Contour (Unified RdYlBu_r with Apple Aesthetics)
        if data_path.exists():
            df = pd.read_csv(data_path)
            X = df[['ph_opt', 'temp_opt']].dropna().values
            y = df.loc[df[['ph_opt', 'temp_opt']].dropna().index, 'log_kcat'].values
            
            if len(X) > 10:
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import Matern, ConstantKernel
                from sklearn.preprocessing import StandardScaler
                from xgboost import XGBRegressor
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
                gp = GaussianProcessRegressor(kernel=kernel, alpha=0.05, normalize_y=True, random_state=42)
                gp.fit(X_scaled, y)
                
                ph_range = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
                temp_range = (X[:, 1].min() - 5, X[:, 1].max() + 5)
                ph_grid = np.linspace(ph_range[0], ph_range[1], 100)
                temp_grid = np.linspace(temp_range[0], temp_range[1], 100)
                PH, TEMP = np.meshgrid(ph_grid, temp_grid)
                conds = np.column_stack([PH.ravel(), TEMP.ravel()])
                Z = gp.predict(scaler.transform(conds)).reshape(PH.shape)
                
                if np.var(Z) < 0.01:
                    xgb = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
                    xgb.fit(X, y)
                    Z = xgb.predict(conds).reshape(PH.shape)
                
                plt.figure(figsize=(11, 8.5))
                z_min, z_max = Z.min(), Z.max()
                levels = np.linspace(z_min, z_max, 50) if z_max - z_min > 1e-4 else np.linspace(z_min-0.5, z_max+0.5, 50)
                
                # Main contour
                cntr = plt.contourf(PH, TEMP, Z, levels=levels, cmap="RdYlBu_r", alpha=0.85)
                cbar = plt.colorbar(cntr, label="Predicted Activity $\log(k_{cat})$")
                cbar.outline.set_visible(False)
                
                # Subtle white lines
                plt.contour(PH, TEMP, Z, levels=15, colors='white', linewidths=0.6, alpha=0.2)
                
                # Global Optimum marker
                opt_ph, opt_temp = optima[ds]
                plt.scatter(opt_ph, opt_temp, color=APPLE_RED, marker="*", s=600, edgecolor="white", 
                            linewidth=2.0, label=f"Discovery Optimum\n(pH {opt_ph:.2f}, {opt_temp:.1f}°C)", zorder=10)
                
                plt.title(f"Kinetic Stability Manifold: {ds.upper()}", fontsize=22, fontweight='bold', pad=30)
                plt.xlabel("pH Level", fontsize=14, fontweight='medium')
                plt.ylabel("Temperature (°C)", fontsize=14, fontweight='medium')
                plt.legend(frameon=True, facecolor="white", shadow=True, borderpad=1, fontsize=12)
                plt.tight_layout()
                plt.savefig(plots_dir / f"{ds}_optimization_contour.png")
                plt.close()

if __name__ == "__main__":
    generate_research_plots()
