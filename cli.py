import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
import yaml
import logging
import sys

from pathlib import Path

# Import functions from main pipeline
from main import setup_logging, load_config, PipelineOrchestrator

app = typer.Typer(help="Enzymology ML Pipeline CLI", add_completion=False)
console = Console()

@app.command()
def run(
    dataset: str = typer.Option("all", "--dataset", "-d", help="Name of the dataset (e.g., 'gst', 'laccase') or 'all'"),
    config_path: str = typer.Option("agent.yaml", "--config", "-c", help="Path to the configuration YAML file")
):
    """
    Run the Enzyme Selection and Bioprocess Optimization Pipeline.
    """
    console.print(Panel.fit("[bold cyan]Enzymology ML Pipeline[/bold cyan]", border_style="cyan"))
    
    # 1. Load config
    config = load_config(config_path)
    if not config:
        console.print("[bold red]Failed to load configuration.[/bold red]")
        sys.exit(1)
        
    # 2. Setup logging
    logger = setup_logging()
    
    # 3. Determine datasets to process
    available_datasets = [ds['name'] for ds in config.get("datasets", {}).get("enzymes", [])]
    
    if dataset.lower() == "all":
        datasets = available_datasets
    else:
        if dataset not in available_datasets:
            console.print(f"[bold red]Dataset '{dataset}' not found in configuration. Available: {', '.join(available_datasets)}[/bold red]")
            sys.exit(1)
        datasets = [dataset]
        
    if not datasets:
        console.print("[bold red]No datasets configured to process.[/bold red]")
        sys.exit(1)
        
    # 4. Process datasets
    orchestrator = PipelineOrchestrator(config, logger)
    for ds in datasets:
        console.print(f"\n[bold yellow]Processing dataset:[/bold yellow] [green]{ds}[/green]")
        try:
            orchestrator.process_dataset(ds)
            console.print(f"[bold green]Successfully processed {ds}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Failed processing {ds}: {str(e)}[/bold red]")
            logger.exception(f"Error processing {ds}")

@app.command()
def infer(
    model: str = typer.Option(..., "--model", "-m", help="Name of the trained model to use (e.g., 'gst', 'laccase', 'lactase')"),
    sequence: str = typer.Option(None, "--sequence", "-s", help="Single amino acid sequence to predict"),
    csv_file: str = typer.Option(None, "--csv", "-f", help="Path to CSV containing a 'sequence' column for batch prediction"),
    output: str = typer.Option(None, "--output", "-o", help="Path to save the output predictions (Default: print to stdout)"),
    format: str = typer.Option("table", "--format", help="Output format: table, csv, json (Default: table)"),
    config_path: str = typer.Option("agent.yaml", "--config", "-c", help="Path to config file")
):
    """
    Run predictions on new sequences using pre-trained models.
    """
    from src.inference.predict import InferenceEngine
    import pandas as pd
    import json
    
    if not sequence and not csv_file:
        console.print("[bold red]You must provide either a --sequence or a --csv file.[/bold red]")
        sys.exit(1)
        
    config = load_config(config_path)
    if not config:
        sys.exit(1)
        
    # Silence most logs for clear CLI output unless debugging
    setup_logging()
    
    console.print(Panel.fit(f"[bold cyan]Enzyme Inference Engine -> {model}[/bold cyan]", border_style="cyan"))
    
    try:
        engine = InferenceEngine(config, model)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize Inference Engine: {str(e)}[/bold red]")
        sys.exit(1)
    
    # Process inputs
    results_df = None
    if sequence:
        res = engine.predict_single(sequence)
        results_df = pd.DataFrame([res])
    else:
        try:
            input_df = pd.read_csv(csv_file)
            results_df = engine.predict_batch(input_df)
        except Exception as e:
            console.print(f"[bold red]Failed to process CSV: {str(e)}[/bold red]")
            sys.exit(1)
            
    # Output handling
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if format.lower() == 'json' or output.endswith('.json'):
            results_df.to_json(output, orient='records', indent=2)
        else: # default to csv
            results_df.to_csv(output, index=False)
        console.print(f"[bold green]Saved predictions to {output}[/bold green]")
        
    if format.lower() == 'table' or not output:
        from rich.table import Table
        
        table = Table(title=f"Predictions ({model})", show_header=True, header_style="bold magenta")
        table.add_column("Rank", justify="center")
        table.add_column("Predicted log(kcat)", justify="right")
        table.add_column("Sequence Start...", justify="left")
        
        for i, row in results_df.head(20).iterrows():
            seq_display = row['sequence'][:30] + "..." if len(row['sequence']) > 30 else row['sequence']
            table.add_row(
                str(i + 1),
                f"{row['predicted_log_kcat']:.3f}",
                seq_display
            )
            
        console.print(table)
        if len(results_df) > 20:
             console.print(f"[italic]... and {len(results_df) - 20} more sequences. Use --output to save them all.[/italic]")

@app.command()
def validate(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Name of the dataset to validate against (e.g., 'gst')"),
    residuals_csv: str = typer.Option(None, "--residuals", "-r", help="Path to save per-sample residuals CSV"),
    config_path: str = typer.Option("agent.yaml", "--config", "-c", help="Path to config file")
):
    """
    Validate a trained model against the original dataset and compute metrics/residuals.
    """
    from src.data.loader import EnzymeDataLoader
    from src.data.preprocessor import EnzymePreprocessor
    from src.inference.predict import InferenceEngine
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import spearmanr
    import numpy as np
    
    config = load_config(config_path)
    if not config:
        sys.exit(1)
        
    setup_logging()
    
    console.print(Panel.fit(f"[bold cyan]Validating Model -> {dataset}[/bold cyan]", border_style="cyan"))
    
    # Load original data
    loader = EnzymeDataLoader(config)
    preprocessor = EnzymePreprocessor(config)
    
    try:
        df = loader.load_dataset(dataset)
        df = preprocessor.preprocess(df, dataset)
    except Exception as e:
        console.print(f"[bold red]Failed to load dataset: {str(e)}[/bold red]")
        sys.exit(1)
        
    if df.empty:
        console.print("[bold red]Dataset is empty after preprocessing.[/bold red]")
        sys.exit(1)
        
    # Init inference engine
    try:
        engine = InferenceEngine(config, dataset)
    except Exception as e:
        console.print(f"[bold red]Failed to load model: {str(e)}[/bold red]")
        sys.exit(1)
        
    console.print(f"[cyan]Running inference on {len(df)} samples...[/cyan]")
    predictions_df = engine.predict_batch(df)
    
    # We must join log_kcat back because predict_batch drops it if it returns sorted/copy
    # Actually predict_batch returns all cols from the input DataFrame
    actual = predictions_df['log_kcat'].values
    predicted = predictions_df['predicted_log_kcat'].values
    
    r2 = r2_score(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    spearman, _ = spearmanr(actual, predicted)
    
    from rich.table import Table
    table = Table(title=f"Validation Metrics (Full Training Set)", show_header=True, header_style="bold green")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")
    
    table.add_row("R² Score", f"{r2:.3f}")
    table.add_row("RMSE", f"{rmse:.3f}")
    table.add_row("MSE", f"{mse:.3f}")
    table.add_row("MAE", f"{mae:.3f}")
    table.add_row("Spearman Rank Corr.", f"{spearman:.3f}")
    
    console.print(table)
    
    console.print("\n[yellow]Note: These metrics represent performance on the full augmented dataset (training accuracy).[/yellow]")
    console.print("[yellow]For strict held-out cross-validation metrics, refer to the generated Markdown reports.[/yellow]")
    
    if residuals_csv:
        predictions_df['residual'] = predictions_df['predicted_log_kcat'] - predictions_df['log_kcat']
        cols = ['uniprot_primary', 'log_kcat', 'predicted_log_kcat', 'residual']
        cols = [c for c in cols if c in predictions_df.columns]
        predictions_df[cols].to_csv(residuals_csv, index=False)
        console.print(f"\n[bold green]Saved residuals to {residuals_csv}[/bold green]")

@app.command()
def export(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Name of the dataset to validate against (e.g., 'gst')"),
    input_csv: str = typer.Option(..., "--input", "-i", help="Path to input sequences CSV"),
    output_prefix: str = typer.Option("export", "--output", "-o", help="Prefix for output files (e.g., 'my_batch' -> 'my_batch_predictions.csv')"),
    config_path: str = typer.Option("agent.yaml", "--config", "-c", help="Path to config file")
):
    """
    Export predictions for a batch of sequences to structured CSV and JSON files.
    """
    from src.inference.predict import InferenceEngine
    from src.reporting.exporter import ResultsExporter
    import pandas as pd
    
    config = load_config(config_path)
    if not config:
        sys.exit(1)
        
    setup_logging()
    
    console.print(Panel.fit(f"[bold cyan]Exporting Predictions -> {dataset}[/bold cyan]", border_style="cyan"))
    
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        console.print(f"[bold red]Failed to load CSV: {str(e)}[/bold red]")
        sys.exit(1)
        
    if 'sequence' not in df.columns:
        console.print("[bold red]CSV must contain a 'sequence' column.[/bold red]")
        sys.exit(1)
        
    try:
        engine = InferenceEngine(config, dataset)
    except Exception as e:
        console.print(f"[bold red]Failed to load model: {str(e)}[/bold red]")
        sys.exit(1)
        
    console.print(f"[cyan]Running inference on {len(df)} samples...[/cyan]")
    predictions_df = engine.predict_batch(df)
    
    actual = predictions_df['predicted_log_kcat'].values
    
    exporter = ResultsExporter(output_dir="outputs/exports")
    exporter.export_predictions_csv(predictions_df, actual, dataset, f"{output_prefix}_predictions.csv", include_sequences=True)
    exporter.export_rankings_csv(predictions_df, f"{output_prefix}_rankings.csv", include_sequences=True)
    
    console.print(f"\n[bold green]Successfully exported results to outputs/exports/[/bold green]")

if __name__ == "__main__":
    app()
