import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ResultsExporter:
    """Exports pipeline results to CSV and JSON formats."""
    
    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_predictions_csv(self, df: pd.DataFrame, predictions: np.ndarray, model_name: str, filename: str, include_sequences: bool = False):
        """Export full predictions table to CSV."""
        export_df = df.copy()
        export_df['predicted_log_kcat'] = predictions
        
        if 'log_kcat' in export_df.columns:
            export_df['residual'] = export_df['predicted_log_kcat'] - export_df['log_kcat']
        
        export_df['rank'] = export_df['predicted_log_kcat'].rank(ascending=False)
        export_df['model_used'] = model_name
        
        # Determine cols to keep
        cols = ['uniprot_primary', 'organism', 'ec', 'sequence_length', 'log_kcat', 'predicted_log_kcat', 'residual', 'rank', 'model_used']
        if include_sequences:
            cols.append('sequence')
            
        cols = [c for c in cols if c in export_df.columns]
        
        out_path = self.output_dir / filename
        export_df[cols].to_csv(out_path, index=False, encoding='utf-8')
        logger.info(f"Exported prediction CSV to {out_path}")
        
    def export_metrics_csv(self, model_results: Dict[str, Any], filename: str):
        """Export model CV metrics to CSV."""
        rows = []
        for model_name, metrics in model_results.items():
            row = {'model_name': model_name}
            for k, v in metrics.items():
                if not isinstance(v, list):  # skip the per-fold lists
                    row[k] = v
            rows.append(row)
            
        metrics_df = pd.DataFrame(rows)
        out_path = self.output_dir / filename
        metrics_df.to_csv(out_path, index=False, encoding='utf-8')
        logger.info(f"Exported metrics CSV to {out_path}")
        
    def export_rankings_csv(self, rankings: pd.DataFrame, filename: str, include_sequences: bool = False):
        """Export ranked enzyme list to CSV."""
        export_df = rankings.copy()
        
        cols = ['rank', 'uniprot_primary', 'organism', 'predicted_log_kcat', 'log_kcat']
        
        if 'log_kcat' in export_df.columns:
             export_df['delta'] = export_df['predicted_log_kcat'] - export_df['log_kcat']
             cols.append('delta')
             
        if include_sequences:
            cols.append('sequence')
            
        cols = [c for c in cols if c in export_df.columns]
        
        out_path = self.output_dir / filename
        export_df[cols].to_csv(out_path, index=False, encoding='utf-8')
        logger.info(f"Exported rankings CSV to {out_path}")
        
    def export_full_results_json(self, metadata: Dict, model_results: Dict, rankings: pd.DataFrame, bioprocess_opt: Dict, feature_importances: Dict, filename: str, include_sequences: bool = False):
        """Export all outcomes to a machine-readable JSON structure."""
        from src.utils.json_utils import json_serializable, clean_dict_nans
        
        rankings_dict = rankings.to_dict(orient='records')
        
        if not include_sequences:
            for row in rankings_dict:
                row.pop('sequence', None)
                
        out_dict = {
            "metadata": clean_dict_nans(metadata),
            "model_performance": clean_dict_nans(model_results),
            "rankings": clean_dict_nans(rankings_dict),
            "bioprocess_optimization": clean_dict_nans(bioprocess_opt) if bioprocess_opt else {},
            "feature_importance": clean_dict_nans(feature_importances) if feature_importances else {}
        }
        
        out_path = self.output_dir / filename
        with open(out_path, 'w') as f:
            json.dump(out_dict, f, default=json_serializable, indent=2)
            
        logger.info(f"Exported full JSON results to {out_path}")
