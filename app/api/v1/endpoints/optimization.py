"""Bioprocess optimization endpoints."""
from fastapi import APIRouter, HTTPException

from app.schemas.requests import BioprocessOptimizationRequest
from app.schemas.responses import OptimizationResponse
from app.services import feature_service, optimization_service, data_service, model_service
from app.utils import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/optimize", response_model=OptimizationResponse, tags=["Optimization"])
async def optimize_conditions(request: BioprocessOptimizationRequest):
    """Optimize pH and temperature conditions for maximum enzyme activity using dataset."""
    try:
        # Get the dataset for this enzyme type
        dataset = data_service.get_dataset(request.dataset_name)
        
        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{request.dataset_name}' not found or not loaded"
            )
        
        # Check if dataset has sufficient pH/temp data
        if 'ph_opt' not in dataset.columns or 'temp_opt' not in dataset.columns:
            raise HTTPException(
                status_code=400,
                detail="Dataset missing pH or temperature data"
            )
        
        # Add scalar features to dataset
        df_with_features = feature_service.feature_engineer.create_features(dataset.copy())
        
        # Get process features
        process_features = feature_service.get_process_features(df_with_features)
        
        if len(process_features) < 2:
            raise HTTPException(
                status_code=400,
                detail="Insufficient process features for optimization"
            )
        
        # Get target values
        y = df_with_features['log_kcat'].values
        
        # Perform optimization
        result = optimization_service.optimize_conditions(
            df_with_features,
            y,
            process_features,
            ph_range=request.ph_range,
            temp_range=request.temp_range
        )
        
        if result.get('skipped'):
            raise HTTPException(
                status_code=400,
                detail=result.get('reason', 'Optimization skipped')
            )
        
        # Get predicted activity for the input sequence using Model A
        if request.enzyme.sequence:
            # Generate features for input enzyme
            input_df = feature_service.enzyme_data_to_dataframe(request.enzyme)
            features, _ = feature_service.generate_features(input_df, request.dataset_name)
            
            # Get model and predict baseline activity
            model = model_service.get_model(request.dataset_name)
            if model:
                baseline_prediction = model_service.predict(model, features)[0]
                result['baseline_log_kcat'] = float(baseline_prediction)
        
        # Remove misleading improvement percentage (Model A and Model B predictions are not comparable)
        result['improvement_percentage'] = None
        
        return OptimizationResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/optimize/heatmap", response_model=OptimizationResponse, tags=["Optimization"])
async def optimize_with_heatmap(request: BioprocessOptimizationRequest):
    """Optimize pH and temperature with visualization heatmap data."""
    try:
        # Get the dataset
        dataset = data_service.get_dataset(request.dataset_name)
        
        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{request.dataset_name}' not found"
            )
        
        # Add features
        df_with_features = feature_service.feature_engineer.create_features(dataset.copy())
        process_features = feature_service.get_process_features(df_with_features)
        
        if len(process_features) < 2:
            raise HTTPException(
                status_code=400,
                detail="Insufficient process features"
            )
        
        y = df_with_features['log_kcat'].values
        
        # Perform optimization
        result = optimization_service.optimize_conditions(
            df_with_features,
            y,
            process_features,
            ph_range=request.ph_range,
            temp_range=request.temp_range
        )
        
        if result.get('skipped'):
            raise HTTPException(
                status_code=400,
                detail=result.get('reason', 'Optimization skipped')
            )
        
        # Generate heatmap data
        ph_range = request.ph_range or tuple(result.get('ph_range', [5.0, 8.0]))
        temp_range = request.temp_range or tuple(result.get('temp_range', [25.0, 70.0]))
        
        heatmap = optimization_service.generate_heatmap(
            df_with_features,
            y,
            process_features,
            ph_range,
            temp_range
        )
        
        if heatmap:
            result['heatmap_data'] = heatmap
        
        return OptimizationResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")
