"""Response schemas for API endpoints."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class PredictionResponse(BaseModel):
    """Response for enzyme activity prediction."""
    
    predicted_log_kcat: float = Field(..., description="Predicted log(kcat) value")
    model_name: str = Field(..., description="Model used for prediction")
    sequence: str = Field(..., description="Input protein sequence")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata (EC, organism, etc.)"
    )


class EnzymeRanking(BaseModel):
    """Single enzyme ranking result."""
    
    rank: int = Field(..., description="Rank position (1 = best)")
    sequence: str = Field(..., description="Protein sequence")
    predicted_log_kcat: float = Field(..., description="Predicted log(kcat)")
    actual_log_kcat: Optional[float] = Field(None, description="Actual log(kcat) if known")
    uniprot_id: Optional[str] = Field(None, description="UniProt ID if available")
    organism: Optional[str] = Field(None, description="Organism name")


class RankingResponse(BaseModel):
    """Response for enzyme ranking."""
    
    rankings: List[EnzymeRanking] = Field(..., description="Ranked enzymes")
    total_enzymes: int = Field(..., description="Total number of enzymes evaluated")
    model_name: str = Field(..., description="Model used for ranking")


class OptimizationResponse(BaseModel):
    """Response for bioprocess optimization."""
    
    optimal_ph: float = Field(..., description="Optimal pH")
    optimal_temperature: float = Field(..., description="Optimal temperature (°C)")
    predicted_log_kcat: float = Field(..., description="Predicted log(kcat) at optimal conditions")
    baseline_log_kcat: Optional[float] = Field(
        None,
        description="Baseline log(kcat) for comparison"
    )
    improvement_percentage: Optional[float] = Field(
        None,
        description="Improvement percentage over baseline"
    )
    ph_range: Optional[List[float]] = Field(
        None,
        description="pH range used for optimization"
    )
    temp_range: Optional[List[float]] = Field(
        None,
        description="Temperature range used for optimization"
    )
    heatmap_data: Optional[Dict[str, Any]] = Field(
        None,
        description="pH-temperature heatmap data for visualization"
    )
    sufficient_variation: bool = Field(
        ...,
        description="Whether data has sufficient variation for optimization"
    )
    n_samples: int = Field(..., description="Number of samples used")


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    
    r2_score: Optional[float] = Field(None, description="R² score")
    rmse: Optional[float] = Field(None, description="Root mean squared error")
    spearman_correlation: Optional[float] = Field(None, description="Spearman correlation")
    mae: Optional[float] = Field(None, description="Mean absolute error")


class ModelInfoResponse(BaseModel):
    """Response for model information."""
    
    model_name: str = Field(..., description="Model identifier")
    dataset_name: str = Field(..., description="Dataset name")
    version: str = Field(..., description="Model version")
    feature_count: int = Field(..., description="Number of features")
    esm_dim: int = Field(..., description="ESM embedding dimensions")
    scalar_features: int = Field(..., description="Number of scalar features")
    metrics: Optional[ModelMetrics] = Field(None, description="Model performance metrics")
    trained_date: Optional[str] = Field(None, description="Training date (ISO format)")
    available: bool = Field(..., description="Whether model is loaded and ready")


class HealthResponse(BaseModel):
    """Response for health check."""
    
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Current timestamp")
    models_loaded: Optional[int] = Field(None, description="Number of models loaded")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
