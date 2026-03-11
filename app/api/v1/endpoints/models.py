"""Model information endpoints."""
from fastapi import APIRouter, HTTPException
from typing import List

from app.schemas.responses import ModelInfoResponse, ModelMetrics
from app.services import model_service
from app.config import settings
from app import __version__

router = APIRouter()


@router.get("/models", response_model=List[ModelInfoResponse], tags=["Models"])
async def list_models():
    """List all available models."""
    models_info = []
    
    for model_name in settings.available_models:
        model = model_service.get_model(model_name)
        info = model_service.get_model_info(model_name)
        
        if model is not None:
            # Try to get model metrics
            metrics = None
            if hasattr(model, 'cv_results') and model.cv_results:
                best_model_metrics = model.cv_results.get(model.best_model, {})
                metrics = ModelMetrics(
                    r2_score=best_model_metrics.get('r2_mean'),
                    spearman_correlation=best_model_metrics.get('spearman_mean'),
                    rmse=best_model_metrics.get('rmse_mean')
                )
            
            # Get feature info - EnzymeSelectionModel stores this differently
            feature_count = 0
            esm_dim = 0
            scalar_features = 0
            
            if hasattr(model, 'feature_names_'):
                feature_names = model.feature_names_
                feature_count = len(feature_names)
                esm_dim = sum(1 for f in feature_names if f.startswith('esm_dim_'))
                scalar_features = feature_count - esm_dim
            
            models_info.append(ModelInfoResponse(
                model_name=model_name,
                dataset_name=model_name,
                version=__version__,
                feature_count=feature_count,
                esm_dim=esm_dim,
                scalar_features=scalar_features,
                metrics=metrics,
                available=True
            ))
        else:
            models_info.append(ModelInfoResponse(
                model_name=model_name,
                dataset_name=model_name,
                version=__version__,
                feature_count=0,
                esm_dim=0,
                scalar_features=0,
                available=False
            ))
    
    return models_info


@router.get("/models/{model_name}", response_model=ModelInfoResponse, tags=["Models"])
async def get_model_info(model_name: str):
    """Get information about a specific model."""
    if model_name not in settings.available_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model = model_service.get_model(model_name)
    
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model '{model_name}' not loaded")
    
    # Get metrics
    metrics = None
    if hasattr(model, 'cv_results') and model.cv_results:
        best_model_metrics = model.cv_results.get(model.best_model, {})
        metrics = ModelMetrics(
            r2_score=best_model_metrics.get('r2_mean'),
            spearman_correlation=best_model_metrics.get('spearman_mean'),
            rmse=best_model_metrics.get('rmse_mean')
        )
    
    # Get feature info
    feature_count = 0
    esm_dim = 0
    scalar_features = 0
    
    if hasattr(model, 'feature_names_'):
        feature_names = model.feature_names_
        feature_count = len(feature_names)
        esm_dim = sum(1 for f in feature_names if f.startswith('esm_dim_'))
        scalar_features = feature_count - esm_dim
    
    return ModelInfoResponse(
        model_name=model_name,
        dataset_name=model_name,
        version=__version__,
        feature_count=feature_count,
        esm_dim=esm_dim,
        scalar_features=scalar_features,
        metrics=metrics,
        available=True
    )
