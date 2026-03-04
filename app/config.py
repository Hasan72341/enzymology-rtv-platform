"""Application configuration."""
from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    app_name: str = "RTV API"
    app_version: str = "1.0.0"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS Settings
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = False
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # Model Settings
    models_dir: Path = Path("outputs/models")
    embeddings_cache_dir: Path = Path("outputs/embeddings")
    config_file: Path = Path("agent.yaml")
    
    # Model names
    available_models: List[str] = ["gst", "laccase", "lactase"]
    
    # ESM-2 Settings (Local Inference)
    esm_backend: str = "pytorch"  # "pytorch" or "onnx"
    esm_model: str = "esm2_t12_35M"
    esm_batch_size: int = 8
    esm_max_sequence_length: int = 1024
    
    # Cache Settings
    enable_embedding_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
