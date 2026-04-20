# Copilot session instructions for Enzymology ML Pipeline

Purpose: Provide concise, repository-specific instructions so future Copilot sessions (and the CLI assistant) can quickly understand how to build, run, test, and reason about this project.

---

## 1) Build, test, and (no-op) lint commands

Local (venv):
- Setup:
  - python3 -m venv env
  - source env/bin/activate
  - pip install -r requirements.txt

Run pipeline / training:
- Full pipeline: `python main.py` (loads `agent.yaml` datasets)
- Single dataset via Typer CLI: `python cli.py run --dataset gst`

Run/inspect inference / CLI tasks:
- Single prediction (CLI):
  - `python cli.py infer --model gst --sequence "MK..."`
- Validate a trained model (CLI):
  - `python cli.py validate --dataset gst`
- Export batch predictions (CLI):
  - `python cli.py export --dataset gst --input sequences.csv --output my_batch`

API server (dev):
- Use venv helper: `./start_api.sh` (ensures env activated; runs uvicorn)
- Or run directly: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

Docker / Compose (recommended for reproducibility):
- Start all services: `docker-compose up -d`
- Run CLI inside container: `docker-compose run cli --dataset laccase`
- Rebuild frontend and start: `docker compose up -d --build frontend`

Tests (repository has simple runnable scripts):
- Run single local embedding test: `python test_local_inference.py`
  - This runs the ESM2 embedder (pytorch backend). It attempts ONNX if onnxruntime installed.
- Run API endpoint smoke tests (start API first): `python test_api.py`
- PyTest: no explicit pytest config provided; if you prefer pytest, install pytest and run `pytest -q <path/to/test_file>.py` for single-file runs.

Linting
- No repo-specific linter commands found. If adding linting, update requirements and CI. If using Docker, rebuild image after updating requirements.txt.

---

## 2) High-level architecture (big picture)

- Three pillars:
  1. `src/` — Machine learning engine: data loaders, feature engineering, embeddings (ESM-2), model wrappers (XGBoost + custom PyTorch FFN ensemble), visualization and reporting.
  2. `app/` — FastAPI backplane exposing model listing and prediction endpoints; intended to load trained weights from `outputs/models`.
  3. `web-frontend/` — React/Vite SPA (served separately); consumes the API for UI features.

- Orchestration:
  - `agent.yaml` defines datasets and ESM provider options. `main.py` implements `PipelineOrchestrator` which runs: load -> preprocess -> feature engineering (ESM2 + scalar features) -> train selection model (Model A) -> optional bioprocess optimization (Model B, Gaussian Process) -> plots and exports.
  - `cli.py` is the Typer CLI wrapper over the orchestrator (commands: run, infer, validate, export).
  - Docker Compose launches three services: `api`, `frontend`, `cli`. The `cli` service mounts the repo so outputs persist to host `outputs/`.

- Data & outputs:
  - Input datasets: CSVs under `data/`, referenced from `agent.yaml`.
  - Outputs: `outputs/models/` (saved model pickle), `outputs/reports/` (JSON/Markdown metrics), `outputs/plots/`, `outputs/exports/` (CSV/JSON predictions).
  - Logs: `logs/pipeline_<timestamp>.log` (created by setup_logging()).

---

## 3) Key conventions & repository-specific patterns

- Config-driven: `agent.yaml` controls datasets, embedding provider (e.g., `provider: local`), and execution parameters. Keep it authoritative.

- Embeddings: `src/features/esm_embeddings.py` supports `provider: local` (PyTorch) and optional remote providers. The embedder accepts `backend` (`pytorch` or `onnx`). The embedder caches to `outputs/` (example: `outputs/test_embeddings`). When changing embedding settings, update `agent.yaml` and consider clearing cache.

- Feature vector composition: feature vectors are built by horizontally concatenating ESM embedding dims (named `esm_dim_0..N`) with scalar/intrinsic features. The embedding output dim is configurable (`output_dim` in config).

- Ensemble / model wrapper patterns:
  - `src/models/enzyme_selection.py` wraps classical (XGBoost/RandomForest) and PyTorch FFN into an sklearn-like interface.
  - `best_model_name` and `best_model` attributes may be present on returned model instances; feature importances are accessible via `model.models['random_forest'].feature_importances_` when available.

- Bioprocess optimization:
  - `src/models/bioprocess_optimization.py` supplies Gaussian Process-based heatmap optimizations for pH/temperature when sufficient samples exist (code checks a minimum sample threshold).

- Tests expectations:
  - `test_api.py` expects the API `/api/v1/models` to return a list (the current test expects 3 entries). If models/datasets change, update tests accordingly.

- Dockerfile note:
  - Dockerfile uses a helper `uv` binary to install dependencies (`uv pip install --system -r requirements.txt`). When adding dependencies, update `requirements.txt` and rebuild images.

- CLI behavior:
  - `python cli.py run --dataset <name>` accepts `all` (processes all configured datasets) or a single dataset defined in `agent.yaml`.
  - `infer`, `validate`, and `export` commands use the same `agent.yaml`-driven model names and expect outputs/models or a trained model to be available.

---

## 4) Files to consult first (most important docs / code paths)
- README.md, QUICKSTART.md, ARCHITECTURE.md, TECHNICAL_DOCUMENTATION.md
- agent.yaml (configuration)
- main.py (PipelineOrchestrator) and cli.py (Typer CLI)
- src/features/esm_embeddings.py (embedding provider/backends)
- src/models/enzyme_selection.py and src/models/bioprocess_optimization.py
- app/main.py and app/api/v1 router(s) for API routes

---

## 5) Short operational checklist for Copilot sessions
- When asked to run or modify pipeline behavior, first check `agent.yaml` for dataset and provider defaults.
- If changing dependencies, update `requirements.txt` and the Dockerfile workflow; rebuild images with Docker Compose.
- For embedding issues, verify whether `provider` is `local` vs remote, and whether `onnxruntime` is installed if ONNX backend is desired.
- When adding tests that hit the API, ensure the API is running locally or mock the endpoints; `test_api.py` is an example of a simple integration smoke test (it calls real endpoints).

---

(End)
