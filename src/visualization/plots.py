"""
Premium Visualization utilities for enzyme selection results.
Features stunning dark-mode aesthetics, neon accents, and clean typography.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Premium Light Aesthetic (White Theme)
BG_COLOR = "#FFFFFF"
PANEL_BG = "#F8FAFC"      # Slate 50
GRID_COLOR = "#E2E8F0"    # Slate 200
PRIMARY_BLUE = "#2563EB"  # Blue 600
PRIMARY_ROSE = "#E11D48"  # Rose 600
PRIMARY_TEAL = "#059669"  # Emerald 600
TEXT_COLOR = "#0F172A"    # Slate 900
SUBTLE_TEXT = "#64748B"   # Slate 500

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "xtick.color": SUBTLE_TEXT,
    "ytick.color": SUBTLE_TEXT,
    "text.color": TEXT_COLOR,
    "font.family": "sans-serif",
    "figure.dpi": 200,
    "savefig.facecolor": BG_COLOR,
    "savefig.edgecolor": BG_COLOR,
})


class EnzymePlotter:
    """Generate breathtaking plots for enzyme selection and optimization."""
    
    def __init__(self, output_dir: str = "outputs/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_enzyme_ranking(
        self, rankings: pd.DataFrame, dataset_name: str, save: bool = True
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        plot_data = rankings.copy()
        plot_data['label'] = plot_data['uniprot_primary'].astype(str) + '\n' + plot_data['organism'].str[:15]
        
        x = np.arange(len(plot_data))
        width = 0.35
        
        # Predicted bars (Neon Cyan)
        bars_pred = ax.bar(
            x - width/2, plot_data['predicted_log_kcat'], width, 
            label='Predicted log(kcat)', color=PRIMARY_BLUE, alpha=0.9,
            edgecolor='white', linewidth=0.5
        )
        
        # Actual bars (Neon Magenta)
        bars_actual = ax.bar(
            x + width/2, plot_data['log_kcat'], width, 
            label='Actual log(kcat)', color=PRIMARY_ROSE, alpha=0.9,
            edgecolor='white', linewidth=0.5
        )
        
        # Styling
        ax.set_xlabel('Enzyme Variant', fontsize=12, fontweight='bold', color=TEXT_COLOR)
        ax.set_ylabel('log(kcat) [s⁻¹]', fontsize=12, fontweight='bold', color=TEXT_COLOR)
        ax.set_title(
            f'Top Enzyme Variants - {dataset_name.upper()}',
            fontsize=16, fontweight='heavy', color="white", pad=20
        )
        ax.set_xticks(x)
        ax.set_xticklabels(plot_data['label'], fontsize=9, rotation=0)
        
        # Legend styling
        legend = ax.legend(fontsize=10, facecolor=PANEL_BG, edgecolor=GRID_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
            
        # Subtle glow effect via spine removal
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color(GRID_COLOR)
        ax.spines['left'].set_color(GRID_COLOR)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f"{dataset_name}_ranking.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved breathtaking ranking plot to {filename}")
        return fig
    
    def plot_ph_temperature_contour(
        self, PH: np.ndarray, TEMP: np.ndarray, predictions: np.ndarray,
        optimal_ph: float, optimal_temp: float, dataset_name: str, save: bool = True
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sleek Magma gradient
        contour = ax.contourf(
            PH, TEMP, predictions, levels=30, cmap='magma', alpha=0.9
        )
        
        # Contour glow line
        contour_lines = ax.contour(
            PH, TEMP, predictions, levels=10, colors=PRIMARY_BLUE, linewidths=0.8, alpha=0.7
        )
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        
        # Mark Optimum securely
        ax.scatter(
            optimal_ph, optimal_temp,
            color=PRIMARY_TEAL, s=250, marker='*',
            edgecolors='white', linewidths=1.5,
            label=f'Optimum\n(pH={optimal_ph:.2f}, T={optimal_temp:.1f}°C)',
            zorder=10
        )
        
        ax.set_xlabel('pH', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Bioprocess Optimization Landscape - {dataset_name.upper()}',
            fontsize=14, fontweight='heavy', color=TEXT_COLOR, pad=15
        )
        
        # Premium Colorbar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Predicted log(kcat) [s⁻¹]', color=TEXT_COLOR, weight='bold')
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_COLOR)
        
        legend = ax.legend(loc='upper right', fontsize=10, facecolor=PANEL_BG, edgecolor=GRID_COLOR)
        plt.setp(legend.get_texts(), color=TEXT_COLOR)
        
        plt.tight_layout()
        if save:
            filename = self.output_dir / f"{dataset_name}_optimization_contour.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved contour plot to {filename}")
        return fig
    
    def plot_feature_importance(self, feature_names: list, importances: np.ndarray, dataset_name: str, save: bool = True) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort features
        indices = np.argsort(importances)[::-1][:15] # Top 15
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        y_pos = np.arange(len(top_features))
        
        # Plot (Neon Teal)
        ax.barh(y_pos, top_importances, align='center', color=PRIMARY_TEAL, alpha=0.9, edgecolor='white', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=10)
        ax.invert_yaxis()  # labels read top-to-bottom
        
        ax.set_xlabel('Relative Importance', fontsize=12, fontweight='bold', color=TEXT_COLOR)
        ax.set_title(f'Top 15 Feature Importances - {dataset_name.upper()}', fontsize=16, fontweight='heavy', color=TEXT_COLOR, pad=20)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color(GRID_COLOR)
        ax.spines['left'].set_color(GRID_COLOR)
            
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f"{dataset_name}_feature_importance.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot: {filename}")
        
        return fig

    def plot_training_loss(self, train_losses: list, val_losses: list, dataset_name: str, save: bool = True, ax=None) -> plt.Figure:
        """Plot training and validation loss curves."""
        save_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            save_fig = save
        else:
            fig = ax.figure
            
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, color=PRIMARY_BLUE, linewidth=2, label='Training Loss', marker='o', markersize=4)
        if val_losses:
            ax.plot(epochs, val_losses, color=PRIMARY_ROSE, linewidth=2, label='Validation Loss', marker='s', markersize=4)
            
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold', color=TEXT_COLOR)
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold', color=TEXT_COLOR)
        ax.set_title(f'Training Dynamics - {dataset_name.upper()}', fontsize=14, fontweight='heavy', color=TEXT_COLOR)
        
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color(GRID_COLOR)
        ax.spines['left'].set_color(GRID_COLOR)
        
        if save_fig:
            plt.tight_layout()
            filename = self.output_dir / f"{dataset_name}_training_loss.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training loss plot: {filename}")
            
        return fig

    def plot_predicted_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, dataset_name: str, save: bool = True, ax=None) -> plt.Figure:
        """Plot predicted vs actual scatter with parity line."""
        save_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            save_fig = save
        else:
            fig = ax.figure
            
        ax.scatter(y_true, y_pred, alpha=0.7, color=PRIMARY_TEAL, edgecolor='white', linewidth=0.5, s=60)
        
        # Parity line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        padding = (max_val - min_val) * 0.05
        ax.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 
                color=LAVENDER, linestyle='--', linewidth=2, label='y = x (Perfect)')
                
        ax.set_xlabel('Actual log(kcat)', fontsize=12, fontweight='bold', color=TEXT_COLOR)
        ax.set_ylabel('Predicted log(kcat)', fontsize=12, fontweight='bold', color=TEXT_COLOR)
        ax.set_title(f'Predicted vs Actual - {model_name}', fontsize=14, fontweight='heavy', color=TEXT_COLOR)
        
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color(GRID_COLOR)
        ax.spines['left'].set_color(GRID_COLOR)
        
        if save_fig:
            plt.tight_layout()
            filename = self.output_dir / f"{dataset_name}_pred_vs_actual.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            
        return fig

    def plot_residual_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, dataset_name: str, save: bool = True, ax=None) -> plt.Figure:
        """Plot distribution of prediction residuals."""
        save_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            save_fig = save
        else:
            fig = ax.figure
            
        residuals = np.array(y_pred) - np.array(y_true)
        
        sns.histplot(residuals, kde=True, ax=ax, color=CORAL, edgecolor='white', linewidth=0.5, alpha=0.7)
        ax.axvline(0, color=SUBTLE_TEXT, linestyle='--', linewidth=2)
        
        ax.set_xlabel('Residual (Predicted - Actual)', fontsize=12, fontweight='bold', color=TEXT_COLOR)
        ax.set_ylabel('Density / Count', fontsize=12, fontweight='bold', color=TEXT_COLOR)
        ax.set_title(f'Residual Analysis - {model_name}', fontsize=14, fontweight='heavy', color=TEXT_COLOR)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color(GRID_COLOR)
        ax.spines['left'].set_color(GRID_COLOR)
        
        if save_fig:
            plt.tight_layout()
            filename = self.output_dir / f"{dataset_name}_residuals.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            
        return fig

    def plot_model_comparison(self, metrics_dict: dict, dataset_name: str, save: bool = True) -> plt.Figure:
        """Plot grouped bar chart comparing models across key metrics."""
        models = list(metrics_dict.keys())
        metrics_to_plot = ['r2', 'rmse', 'mae', 'spearman']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        width = 0.2
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
        colors = [PRIMARY_BLUE, PRIMARY_ROSE, PRIMARY_TEAL, SUBTLE_TEXT]
        
        for i, metric in enumerate(metrics_to_plot):
            values = [metrics_dict[m].get(metric, 0) for m in models]
            ax.bar(x + offsets[i], values, width, label=metric.upper(), color=colors[i], alpha=0.9, edgecolor='white', linewidth=0.5)
            
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylabel('Metric Score', fontsize=12, fontweight='bold', color=TEXT_COLOR)
        ax.set_title(f'Model Comparison - {dataset_name.upper()}', fontsize=16, fontweight='heavy', color=TEXT_COLOR, pad=20)
        
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc='center left', bbox_to_anchor=(1, 0.5))
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color(GRID_COLOR)
        ax.spines['left'].set_color(GRID_COLOR)
        
        plt.tight_layout()
        if save:
            filename = self.output_dir / f"{dataset_name}_model_comparison.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison plot: {filename}")
            
        return fig
        
    def generate_model_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray, metrics_dict: dict, best_model_name: str, dataset_name: str) -> plt.Figure:
        """Generate a composite diagnostics 2x2 grid."""
        fig = plt.figure(figsize=(16, 12))
        
        # We'll use gridspec for layout
        gs = fig.add_gridspec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, :]) # Top row: model comparison
        # Since plot_model_comparison creates a new figure, we need to adapt it or manually plot here. 
        # For simplicity, we just use the grouped bar chart logic directly on ax1
        models = list(metrics_dict.keys())
        metrics_to_plot = ['r2', 'rmse', 'mae', 'spearman']
        x = np.arange(len(models))
        width = 0.2
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
        colors = [PRIMARY_BLUE, PRIMARY_ROSE, PRIMARY_TEAL, SUBTLE_TEXT]
        
        for i, metric in enumerate(metrics_to_plot):
            values = [metrics_dict[m].get(metric, 0) for m in models]
            ax1.bar(x + offsets[i], values, width, label=metric.upper(), color=colors[i], alpha=0.9, edgecolor='white', linewidth=0.5)
            
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontsize=11)
        ax1.set_title('Cross-Validation Model Comparison', fontsize=14, fontweight='heavy', color=TEXT_COLOR)
        ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
        for spine in ['top', 'right']: ax1.spines[spine].set_visible(False)
        
        # Bottom Left: Pred vs Actual
        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_predicted_vs_actual(y_true, y_pred, best_model_name, dataset_name, save=False, ax=ax2)
        
        # Bottom Right: Residuals
        ax3 = fig.add_subplot(gs[1, 1])
        self.plot_residual_distribution(y_true, y_pred, best_model_name, dataset_name, save=False, ax=ax3)
        
        plt.tight_layout()
        filename = self.output_dir / f"{dataset_name}_model_diagnostics.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved diagnostic composite figure to {filename}")
        
        return fig
