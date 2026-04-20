# FastAPI Integration & Endpoints

## Overview
The backend is driven by a high-performance **FastAPI** service (`app/main.py`). It handles the instantiation of the pre-trained machine learning models, dynamically computes ESM-2 embeddings on the fly, and serves inferences to the React frontend.

## Core Endpoints

### 1. Prediction & Ranking
- `POST /api/v1/predict/single`
  - **Payload:** Takes a single enzyme sequence and optional metadata (EC, Organism, $K_m$, MW, pH/Temp Opt).
  - **Response:** Returns the predicted $\log(k_{cat})$.
- `POST /api/v1/predict/batch`
  - **Payload:** An array of enzyme sequences.
  - **Response:** An array of predictions, useful for unranked bulk processing.
- `POST /api/v1/rank`
  - **Payload:** An array of enzyme sequences.
  - **Response:** An ordered, ranked list of enzymes sorted by their predicted $\log(k_{cat})$. Returns a computed `rank` integer for strict UI rendering.

### 2. Bioprocess Optimization
- `POST /api/v1/optimize`
  - **Payload:** A single enzyme variant.
  - **Response:** Returns the optimal pH and temperature parameters.
- `POST /api/v1/optimize/heatmap`
  - **Payload:** A single enzyme variant.
  - **Response:** Returns the optimal parameters alongside a dense 2D grid matrix of `predictions`, `ph_grid`, and `temp_grid`. This data is directly consumed by the frontend's Plotly.js component to render the topological stability manifold.

### 3. Model Diagnostics & Health
- `GET /api/v1/models`
  - **Response:** A list of active, loaded dataset models (e.g., `gst`, `laccase`, `lactase`).
- `GET /api/v1/models/{model_name}`
  - **Response:** Returns strict empirical validation metrics (R² Score, RMSE, Spearman Correlation, MAE) and internal architecture details (ESM dimensionality, feature counts) to populate the Case Studies UI.
- `GET /api/v1/health` & `GET /api/v1/ready`
  - **Response:** System status checks ensuring the transformer weights and model tensors are securely loaded in memory before allowing client connections.