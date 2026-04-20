# Enzymology ML Pipeline: Comprehensive Technical Documentation

## 1. Project Overview & Objectives
The **Enzymology ML Pipeline** is a containerized, full-stack biological machine learning application built to predict the intrinsic catalytic efficiency (`log_kcat`) and optimal bioprocess conditions (pH, Temperature) for industrial enzyme variants from their raw amino acid sequences. 

By unifying state-of-the-art biological language models (**ESM-2**) with heavily mathematically-regularized classical and deep learning frameworks, this platform reliably predicts and ranks enzyme performance across unseen datasets, even when constrained by extremely small industrial measurement samples.

This end-to-end suite includes:
- A PyTorch / Scikit-Learn **Machine Learning Engine**.
- A FastAPI **Backend Service**.
- A React / Vite **Web Interface**.

---

## 2. System Architecture

The architecture separates responsibilities strictly across functional layers, maintaining absolute modularity and making it highly suitable for zero-configuration Docker scaling.

### Directory Structure & Layers
```text
Enzymology 2/
├── docker-compose.yml       # Central Orchestration (api, frontend, cli)
├── agent.yaml               # Pipeline execution configuration (datasets)
├── cli.py                   # Typer CLI Entrypoint for triggering ML Runs
├── main.py                  # Core PipelineOrchestrator Logic
├── web-frontend/            # React.js SPA Client with Vite 
└── src/                     # Core Business and Engine Logic
    ├── data/                # loader.py (Parses CSV) & preprocessor.py  
    ├── features/            # esm_embeddings.py & scalar_features.py
    ├── models/              # hybrid_nn.py, hyperparameter_search.py, enzyme_selection.py
    ├── visualization/       # plots.py (Contour & Feature ranking generations)
    └── reporting/           # report_generator.py (Markdown report logic)
```

---

## 3. The Machine Learning Engine

The core predictive engine operates within `src/models/` and sequentially performs feature extraction, heuristic hyperparameter optimization, and ensemble training.

### A. Feature Extraction
For every input sequence, the pipeline generates a unified **274-dimensional vector**:
1. **ESM-2 Biological Sequences:** An API connection to the NVIDIA NIM Cloud fetches a robust 256-dimensional mean-pooled latent biological representation using the 650 million parameter `esm2_t33_650M_UR50D` language model.
2. **Scalar Features:** Physiochemical constants (pH_opt, temp_opt, molecular weight, Km) are concatenated alongside the embeddings as native regression features.

### B. The Hybrid Ensemble Model
Because modeling continuous enzyme data limits the maximum samples available to `n < 300` per class, the system uses an ensemble to maximize predictive power:
- **XGBoost Regressor:** Optimized for identifying distinct, non-linear scalar interactions (like optimal pH and exact temperature splits).
- **PyTorch Residual FFN (`hybrid_nn.py`):** A custom neural network built with Multi-layer Perceptron (MLP) blocks, dynamically scaling hidden units and skip-connections for deep biological manifold mapping.

### C. Automated Data Augmentation & Regularization
To combat overfitting on tiny, sparse biological datasets, the pipeline forcibly expands the manifolds across both classical and gradient-descent algorithms:
1. **Dynamic MixUp:** Random batches dynamically interpolate pairs of enzymes and their corresponding `log_kcat` targets mid-epoch inside the PyTorch dataloader.
2. **Gaussian Injection:** The MLP blocks utilize a mathematical entry layer that perturbs input features by $\sigma = 0.02$, teaching the model to ignore biological noise.
3. **Tabular SMOTE:** During Random Forest and XGBoost Cross-Validation splits, the algorithm generates completely synthetic linear data interpolations, strictly doubling the exact fold size just before `model.fit()`.

### D. Hyperparameter Routing
Before execution, `hyperparameter_search.py` summons **Optuna** to execute a mathematical randomized search using internal validation folds to mathematically derive the ultimate network width, learning rate, and tree depth.

---

## 4. Sub-Models Processing

Depending on the targeted industrial application, the pipeline branches into two distinctive execution nodes:

- **Model A (Enzyme Selection):** Evaluates 100% of candidates based strictly on sequence similarity via the ensemble architectures, ranking the absolute theoretically best enzyme candidates directly out of the pool.
- **Model B (Bioprocess Optimization):** Employs fundamentally different logic, pivoting to a mathematical **Gaussian Process Regressor**. By sampling the predicted data over varying theoretical 2D planes, it produces a dynamic topographical contour mapping the exact ideal pH / Temperature condition that mathematically peaks the variant's potential.

---

## 5. Metrics, Diagnostics & Outputs

Following execution, the system dynamically analyzes absolute model competence over the mathematical evaluation structures safely bridging performance and interpretation:

### A. Core Validation Metrics Tracker
The cross-validation pipeline natively records and compares an array of advanced statistical boundaries natively stored per fold:
- `R²` (Coefficient of Determination), `RMSE` (Root Mean Squared Error)
- `MSE` (Mean Squared Error), `MAE` (Mean Absolute Error)
- `Spearman` Rank-Order Coefficient

### B. Analytical Visualization Engine
`src/visualization/plots.py` utilizes heavy `seaborn-v0_8-whitegrid` styling rendering four publication-ready analytic architectures:
1. **Model Comparison Grid**: Correlates the error and performance distributions using comparative radar/bar visualizations across XGBoost, PyTorch, and RF.
2. **Scatter Parity Plots**: Validates actual `log_kcat` metrics mapping over sequence predictions using an identical $y=x$ alpha overlay.
3. **Residual Kernel Density**: Projects deviations via normalized histogram distributions assessing unbiased predictive scaling.
4. **Training Dynamics Tracker**: Renders standard Hybrid Ensemble FFN loss convergence directly across epoch trajectories.

### C. The Exporter Protocol
The system ensures continuous downstream programmatic utility via `src/reporting/exporter.py`:
- Saves continuous metrics to `outputs/reports/{dataset}_metrics.csv`.
- Dumps sequences mapped with the ensemble prediction values via full metadata arrays into `outputs/exports/`.
- Compiles rigorous `_validation_metrics.json` configurations combining feature parameters with pipeline hashing logic.

---

## 6. Web and Infrastructure Deployment

### Environment
The entire software operates perfectly offline without localized Python dependencies thanks to Docker Compose. There are zero unmapped artifacts.

#### `docker-compose.yml` Services
- `api` (Port 8000): A FastAPI/Uvicorn server hosting asynchronous REST endpoints to list trained models directly from `outputs/models`.
- `frontend` (Port 5173): A React SPA capable of communicating dynamically via Vite proxies directly to the backend. Rebuild using `docker compose up -d --build frontend` to fetch environment state logic.
- `cli` (Ephemeral): Handles the heavy-duty ML scripts securely mounting `.:/app` volumes. 

### Triggering the AI
The Typer CLI located in `cli.py` exposes three strictly isolated commands for executing the machine learning pipeline tasks safely onto `outputs`:

1. **Training Engine**: Spawns datasets mapped in `agent.yaml` to dynamically build neural models.
   ```bash
   python cli.py run --dataset all
   ```
2. **Inference Validator**: Loads standalone CSV arrays analyzing external data without initiating gradient-descent structures parsing validation logic reliably.
   ```bash
   python cli.py validate --dataset gst
   ```
3. **Batch Exporter**: Utilizes pre-trained algorithms routing unknown feature columns directly to JSON and CSV endpoints inside `outputs/exports`.
   ```bash
   python cli.py export --dataset gst --input custom_sequences.csv
   ```
