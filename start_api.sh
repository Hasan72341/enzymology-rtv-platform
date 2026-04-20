#!/bin/bash
# Start the FastAPI server

echo "Starting Enzyme Activity Prediction API..."
echo "========================================="
echo ""

# Change to project directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "env" ]; then
    echo "Virtual environment not found!"
    echo "Please create and activate a virtual environment first:"
    echo "  python -m venv env"
    echo "  source env/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source env/bin/activate
fi

# Start the server
echo "Starting server on http://localhost:8000"
echo "API documentation will be available at:"
echo "  - Swagger UI: http://localhost:8000/docs"
echo "  - ReDoc: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
