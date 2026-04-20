# Quick Start Guide

## Installation

```bash
cd /Users/hasan/Documents/Enzymology

# Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# Install core dependencies
pip install pandas numpy scikit-learn xgboost pyyaml matplotlib seaborn scipy tqdm

# For local ESM-2 (optional)
pip install torch transformers
```

## Configuration

Edit `agent.yaml` to set ESM-2 provider:
```yaml
provider: local  # Start with this
```

## Run

```bash
python main.py
```

## Check Results

```bash
ls outputs/reports/     # Markdown reports
ls outputs/plots/       # Visualizations
ls outputs/models/      # Trained models
```

See `walkthrough.md` for full documentation.
