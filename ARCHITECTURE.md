# 🏗 Architecture & Design

This project contains three major pillars engineered for separation of concerns, scalability, and ease of deployment.

## 1. Machine Learning Orchestration (`src/` and `main.py`)

- **`main.py` / `cli.py`**: The central nervous system. `cli.py` exposes a Type/Rich CLI interface to `main.py`, which is implemented as an Object-Oriented Orchestrator (`PipelineOrchestrator`).
- **`src/data/`**: Manages data ingestion pipelines and preprocessors. Data constraints are governed by `agent.yaml`.
- **`src/features/`**:
  - `esm_embeddings.py`: Makes dynamic calls to local or server-side (NVIDIA NIM) large language models to embed amino-acid sequences into powerful mathematical forms.
  - `scalar_features.py`: Cleans and encodes pH, temperature, and measurements.
- **`src/models/`**:
  - `enzyme_selection.py`: The wrapper for evaluating multiple models.
  - `hybrid_nn.py`: **[INNOVATION]** A unified module integrating a custom PyTorch Feed Forward sequence parser with a traditional robust tree (XGBoost) into a single scikit-learn compatible object.
- **`src/visualization/`**: Generates high-resolution `.png` visuals incorporating a premium glassmorphic dark-mode configuration, using `matplotlib` patches cleanly to emulate neon glows.

## 2. RTV API Backplane (`app/`)
- A strict schema-driven FastAPI layer.
- Instantiates trained model weights into a live server environment memory to serve real-time asynchronous inference via the Route defined in `app/api/v1/router.py`.

## 3. Web Frontend (`web-frontend/`)
- A modularized Vue / React component structure (using Vite). 
- Pulls data from `app` dynamically to serve business users directly without terminal knowledge.

## 4. Container Management Layers (`Docker`)
- Completely integrated. The root `docker-compose.yml` launches independent volumes for the `API`, `Frontend`, and `CLI` components connecting them synchronously via bridged networks. Dev workflows are fully enabled.
