#!/usr/bin/env python3
"""
Main orchestration script for enzyme selection and bioprocess optimization pipeline.
Refactored for modularity, strict typing, and object-oriented execution.
"""
import yaml
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import EnzymeDataLoader
from src.data.preprocessor import EnzymePreprocessor
from src.reporting.exporter import ResultsExporter
from src.features.esm_embeddings import ESM2Embedder
from src.features.scalar_features import ScalarFeatureEngineer
from src.models.enzyme_selection import EnzymeSelectionModel
from src.models.bioprocess_optimization import BioprocessOptimizer
from src.visualization.plots import EnzymePlotter
from src.reporting.report_generator import ReportGenerator

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure structured logging outputs."""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f"pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized to {log_file}")
    return logger


def load_config(config_path: str = "agent.yaml") -> Dict[str, Any]:
    """Load and return the configuration dict."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class PipelineOrchestrator:
    """Manages the lifecycle steps for Enzyme prediction and optimization."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        self.loader = EnzymeDataLoader(config)
        self.preprocessor = EnzymePreprocessor(config)
        self.feature_eng = ScalarFeatureEngineer(config.get('feature_engineering', {}))
        self.embedder = ESM2Embedder(config.get('feature_engineering', {}))
        self.plotter = EnzymePlotter()
        self.reporter = ReportGenerator()
        self.exporter = ResultsExporter()

    def load_and_preprocess(self, dataset_name: str) -> Optional[pd.DataFrame]:
        self.logger.info("Step 1 & 2: Loading & Preprocessing Data")
        df = self.loader.load_dataset(dataset_name)
        df = self.preprocessor.preprocess(df, dataset_name)
        if df.empty:
            self.logger.warning(f"Dataset {dataset_name} empty after preprocess.")
            return None
        return df

    def feature_engineering(self, df: pd.DataFrame, dataset_name: str) -> Tuple[np.ndarray, List[str]]:
        self.logger.info("Step 3: Feature Engineering")
        df = self.feature_eng.create_features(df)
        embeddings = self.embedder.embed_sequences(df, dataset_name)
        intrinsic_features = self.feature_eng.get_intrinsic_features(df)
        X_intrinsic = df[intrinsic_features].values
        
        X = np.hstack([embeddings, X_intrinsic])
        feature_names = [f'esm_dim_{i}' for i in range(embeddings.shape[1])] + intrinsic_features
        return X, feature_names

    def run_enzyme_selection(self, df: pd.DataFrame, X: np.ndarray, feature_names: List[str], dataset_name: str):
        self.logger.info("Step 4: Training Selection Model A")
        y = df['log_kcat'].values
        groups = df['ec'].factorize()[0]
        sample_weights = df['n_measurements'].values
        
        model = EnzymeSelectionModel(self.config['models']['enzyme_selection'])
        results = model.train(X, y, groups, sample_weights, feature_names)
        
        y_pred = model.predict(X)
        top_enzymes = model.rank_enzymes(df, y_pred)
        
        # Plot model diagnostics
        best_model_name = getattr(model, 'best_model_name', list(results.keys())[0] if results else "Unknown")
        self.plotter.generate_model_diagnostics(y, y_pred, results, best_model_name, dataset_name)
        
        model_path = Path("outputs/models") / f"{dataset_name}_selection_model.pkl"
        model_path.parent.mkdir(exist_ok=True)
        model.save(str(model_path))
        
        # Save validation metrics to JSON
        metrics_path = Path("outputs/reports") / f"{dataset_name}_validation_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        from src.utils.json_utils import json_serializable
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=4, default=json_serializable)
        
        return model, results, top_enzymes

    def run_bioprocess_optimization(self, df: pd.DataFrame, dataset_name: str):
        self.logger.info("Step 5: Bioprocess Optimization Analysis Model B")
        y = df['log_kcat'].values
        process_features = self.feature_eng.get_process_features(df)
        min_samples = 20
        n_samples = len(df)
        
        if len(process_features) >= 2 and n_samples >= min_samples:
            X_process = df[process_features[:2]].values
            optimizer = BioprocessOptimizer(self.config['models']['bioprocess_optimization'])
            
            optimizer.train(X_process, y, model_type='gaussian_process')
            ph_opt_b = df['ph_opt'].median() if 'ph_opt' in df.columns else None
            t_opt_b = df['temp_opt'].median() if 'temp_opt' in df.columns else None
            
            opt_ph, opt_temp, pred_act = optimizer.optimize(df, ph_opt_b, t_opt_b)
            heatmap_data = optimizer.predict_heatmap(
                (df[process_features[0]].min(), df[process_features[0]].max()),
                (df[process_features[1]].min(), df[process_features[1]].max())
            )
            
            return {
                'ph': opt_ph, 'temperature': opt_temp, 'predicted_log_kcat': pred_act,
                'ph_opt_baseline': ph_opt_b, 'temp_opt_baseline': t_opt_b,
                'baseline_log_kcat': df['log_kcat'].median()
            }, heatmap_data
        
        self.logger.warning(f"Insufficient data for optimization: {n_samples} < {min_samples}")
        return None, (None, None, None)

    def process_dataset(self, dataset_name: str) -> None:
        self.logger.info(f"\n{'='*60}\nProcessing dataset: {dataset_name.upper()}\n{'='*60}")
        
        # Steps 1-3
        df = self.load_and_preprocess(dataset_name)
        if df is None: return
        X, feature_names = self.feature_engineering(df, dataset_name)
        
        # Step 4
        sel_model, sel_results, top_enzymes = self.run_enzyme_selection(df, X, feature_names, dataset_name)
        
        # Step 5
        opt_conditions, heatmap_data = self.run_bioprocess_optimization(df, dataset_name)
        
        # Step 6: Visualizations
        self.logger.info("Step 6: Generating visualizations")
        self.plotter.plot_enzyme_ranking(top_enzymes, dataset_name)
        if opt_conditions and heatmap_data[0] is not None:
            PH, TEMP, pred_heatmap = heatmap_data
            self.plotter.plot_ph_temperature_contour(PH, TEMP, pred_heatmap, opt_conditions['ph'], opt_conditions['temperature'], dataset_name)
            
        # Plot feature importance from the best model if it supports it
        best_model = getattr(sel_model, 'best_model', None)
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            self.plotter.plot_feature_importance(feature_names, importances, dataset_name)
        elif 'random_forest' in sel_model.models:
            importances = sel_model.models['random_forest'].feature_importances_
            self.plotter.plot_feature_importance(feature_names, importances, dataset_name)
            
        # Step 7: Reports
        self.logger.info("Step 7: Generating final report")
        dataset_info = {
            'ec_number': df.get('ec_number', pd.Series(['N/A'])).iloc[0],
            'total_samples': len(df),
            'unique_enzymes': df['uniprot_primary'].nunique(),
            'unique_organisms': df['organism'].nunique() if 'organism' in df.columns else 0,
            'log_kcat_min': df['log_kcat'].min() if 'log_kcat' in df.columns else 0,
            'log_kcat_max': df['log_kcat'].max() if 'log_kcat' in df.columns else 0,
            'real_world_problem': df['real_world_problem'].iloc[0] if 'real_world_problem' in df.columns else 'N/A',
        }
        
        # Enrich opt_conditions with experimental variation ranges for accurate reporting
        report_opt_conditions = dict(opt_conditions) if opt_conditions else {}
        if opt_conditions:
            report_opt_conditions['ph_range'] = (
                df['ph_opt'].max() - df['ph_opt'].min()
            ) if 'ph_opt' in df.columns and df['ph_opt'].notna().sum() > 1 else 0
            report_opt_conditions['temp_range'] = (
                df['temp_opt'].max() - df['temp_opt'].min()
            ) if 'temp_opt' in df.columns and df['temp_opt'].notna().sum() > 1 else 0
        
        report_path = self.reporter.generate_report(
            dataset_name, dataset_info, sel_results, top_enzymes, report_opt_conditions
        )

        
        # Determine best model name
        best_model_name = getattr(sel_model, 'best_model_name', list(sel_results.keys())[0] if sel_results else "Unknown")
        y_pred = sel_model.predict(X)
        
        # Export machine-readable results
        self.logger.info("Step 7: Exporting Data Files (JSON/CSV)")
        self.exporter.export_predictions_csv(df, y_pred, best_model_name, f"{dataset_name}_full_predictions.csv", include_sequences=False)
        self.exporter.export_metrics_csv(sel_results, f"{dataset_name}_metrics.csv")
        self.exporter.export_rankings_csv(top_enzymes, f"{dataset_name}_top_enzymes.csv", include_sequences=False)
        
        # Note: In a real robust system, we would grab feature importances from the model
        feature_importances = getattr(sel_model.best_model, 'feature_importances_', None)
        fi_dict = {}
        if feature_importances is not None and len(feature_names) == len(feature_importances):
            fi_dict = {f: float(i) for f, i in zip(feature_names, feature_importances)}
            
        metadata = {
            "dataset_name": dataset_name,
            "n_samples": len(df),
            "n_features": len(feature_names),
        }
        
        self.exporter.export_full_results_json(
            metadata=metadata,
            model_results=sel_results,
            rankings=top_enzymes,
            bioprocess_opt=opt_conditions,
            feature_importances=fi_dict,
            filename=f"{dataset_name}_full_results.json"
        )
        
        self.logger.info(f"Pipeline complete for {dataset_name}.")
        return {
            'model': sel_model,
            'results': sel_results,
            'optimal_conditions': opt_conditions,
            'report_path': report_path
        }


def main():
    logger = setup_logging()
    config = load_config()
    np.random.seed(config.get('execution', {}).get('reproducibility', {}).get('random_seed', 42))
    
    datasets = [ds['name'] for ds in config['datasets']['enzymes']]
    orchestrator = PipelineOrchestrator(config, logger)
    
    for dataset_name in datasets:
        try:
            orchestrator.process_dataset(dataset_name)
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
