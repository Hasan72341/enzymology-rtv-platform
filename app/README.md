# Enzyme Activity Prediction API

FastAPI application for enzyme activity prediction and bioprocess optimization.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the API

```bash
# Start the server
uvicorn app.main:app --reload --port 8000

# Or with custom host/port
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Checks
- `GET /health` - Basic health check
- `GET /ready` - Readiness check (models loaded)

### Models
- `GET /api/v1/models` - List all available models
- `GET /api/v1/models/{model_name}` - Get specific model info

### Predictions
- `POST /api/v1/predict/single` - Predict activity for single enzyme
- `POST /api/v1/predict/batch` - Predict activity for multiple enzymes
- `POST /api/v1/rank` - Rank enzymes by predicted activity

### Optimization
- `POST /api/v1/optimize` - Optimize pH and temperature
- `POST /api/v1/optimize/heatmap` - Optimize with heatmap data

## Example Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/single" \
  -H "Content-Type: application/json" \
  -d '{
    "enzyme": {
      "sequence": "MKALSKLKAEEGIWMTDVPVPELGHN...",
      "ec": "1.1.1.1",
      "organism": "Homo sapiens",
      "n_measurements": 1
    },
    "dataset_name": "gst"
  }'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "enzymes": [
      {
        "sequence": "MKALSKLK...",
        "ec": "1.1.1.1",
        "organism": "Homo sapiens"
      },
      {
        "sequence": "MANEVIK...",
        "ec": "1.1.1.1",
        "organism": "Mus musculus"
      }
    ],
    "dataset_name": "gst"
  }'
```

## Configuration

Create a `.env` file to override default settings:

```env
# API Settings
DEBUG=false
HOST=0.0.0.0
PORT=8000

# CORS
CORS_ORIGINS=["http://localhost:3000"]

# Models
USE_NVIDIA_NIM=false
ESM_BATCH_SIZE=8

# Logging
LOG_LEVEL=INFO
```

## Available Models

- `gst` - Glutathione S-transferase
- `laccase` - Laccase
- `lactase` - Lactase

Models are automatically loaded from `outputs/models/` on startup.

## Development

### Project Structure

```
app/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration settings
├── api/
│   └── v1/
│       ├── router.py    # API v1 router
│       └── endpoints/   # API endpoints
├── schemas/             # Pydantic models
│   ├── requests.py      # Request schemas
│   └── responses.py     # Response schemas
├── services/            # Business logic
│   ├── model_service.py
│   ├── feature_service.py
│   └── optimization_service.py
└── utils/               # Utilities
    └── logger.py
```

## Notes

- ESM-2 embedding generation may take 5-30 seconds depending on sequence length
- Embeddings are cached in `outputs/embeddings/` for faster subsequent requests
- All models are loaded into memory on startup (~1-2GB RAM)
- Request timeout recommended: 60 seconds for single predictions, 120 seconds for batch
