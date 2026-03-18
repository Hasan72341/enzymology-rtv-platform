FROM python:3.10-slim

# Copy uv static binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
# Allow setting a proper mirror via build arg (e.g. Aliyun or Tsinghua mirror)
ARG PYPI_MIRROR=https://pypi.org/simple
ENV UV_INDEX_URL=${PYPI_MIRROR}

COPY requirements.txt .
# Use uv for extremely fast dependency setup
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . .

# Expose both FastAPI and potentially Vite local ports if needed
EXPOSE 8000
EXPOSE 5173

# Default behavior: run the training/orchestration main
CMD ["python", "main.py"]
