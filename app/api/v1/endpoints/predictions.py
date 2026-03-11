"""Prediction endpoints."""
from fastapi import APIRouter, HTTPException
from typing import List

from app.schemas.requests import EnzymePredictionRequest, BatchPredictionRequest
from app.schemas.responses import PredictionResponse, RankingResponse, EnzymeRanking
from app.services import model_service, feature_service
from app.utils import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/predict/single", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(request: EnzymePredictionRequest):
    """Predict enzyme activity for a single sequence."""
    # Get model
    model = model_service.get_model(request.dataset_name)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{request.dataset_name}' is not available"
        )
    
    try:
        # Convert to DataFrame
        df = feature_service.enzyme_data_to_dataframe(request.enzyme)
        
        # Generate features
        features, feature_names = feature_service.generate_features(df, request.dataset_name)
        
        # Predict
        prediction = model_service.predict(model, features)[0]
        
        # Prepare metadata
        metadata = {
            "ec": request.enzyme.ec,
            "organism": request.enzyme.organism,
            "n_measurements": request.enzyme.n_measurements,
            "kcat_std": request.enzyme.kcat_std
        }
        if request.enzyme.kmValue:
            metadata["kmValue"] = request.enzyme.kmValue
        if request.enzyme.ph_opt:
            metadata["ph_opt"] = request.enzyme.ph_opt
        if request.enzyme.temp_opt:
            metadata["temp_opt"] = request.enzyme.temp_opt
        
        return PredictionResponse(
            predicted_log_kcat=float(prediction),
            model_name=request.dataset_name,
            sequence=request.enzyme.sequence,
            metadata=metadata
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=List[PredictionResponse], tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """Predict enzyme activity for multiple sequences."""
    # Get model
    model = model_service.get_model(request.dataset_name)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{request.dataset_name}' is not available"
        )
    
    try:
        # Convert to DataFrame
        df = feature_service.enzymes_list_to_dataframe(request.enzymes)
        
        # Generate features
        features, feature_names = feature_service.generate_features(df, request.dataset_name)
        
        # Predict
        predictions = model_service.predict(model, features)
        
        # Build responses
        responses = []
        for idx, enzyme_data in enumerate(request.enzymes):
            metadata = {
                "ec": enzyme_data.ec,
                "organism": enzyme_data.organism,
                "n_measurements": enzyme_data.n_measurements,
                "kcat_std": enzyme_data.kcat_std
            }
            if enzyme_data.kmValue:
                metadata["kmValue"] = enzyme_data.kmValue
            if enzyme_data.ph_opt:
                metadata["ph_opt"] = enzyme_data.ph_opt
            if enzyme_data.temp_opt:
                metadata["temp_opt"] = enzyme_data.temp_opt
            
            responses.append(PredictionResponse(
                predicted_log_kcat=float(predictions[idx]),
                model_name=request.dataset_name,
                sequence=enzyme_data.sequence,
                metadata=metadata
            ))
        
        return responses
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post("/rank", response_model=RankingResponse, tags=["Predictions"])
async def rank_enzymes(request: BatchPredictionRequest):
    """Rank enzymes by predicted activity."""
    # Get model
    model = model_service.get_model(request.dataset_name)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{request.dataset_name}' is not available"
        )
    
    try:
        # Convert to DataFrame
        df = feature_service.enzymes_list_to_dataframe(request.enzymes)
        
        # Generate features
        features, feature_names = feature_service.generate_features(df, request.dataset_name)
        
        # Predict
        predictions = model_service.predict(model, features)
        
        # Add predictions to DataFrame
        df['predicted_log_kcat'] = predictions
        
        # Rank
        top_k = min(len(df), 20)  # Return top 20 or fewer
        ranked_df = model_service.rank_enzymes(model, df, predictions, top_k=top_k)
        
        # Build ranking response
        rankings = []
        for _, row in ranked_df.iterrows():
            rankings.append(EnzymeRanking(
                rank=int(row.get('rank', 0)),
                sequence=row.get('sequence', ''),
                predicted_log_kcat=float(row.get('predicted_log_kcat', 0.0)),
                actual_log_kcat=float(row['log_kcat']) if 'log_kcat' in row and row['log_kcat'] is not None else None,
                uniprot_id=row.get('uniprot_primary'),
                organism=row.get('organism')
            ))
        
        return RankingResponse(
            rankings=rankings,
            total_enzymes=len(df),
            model_name=request.dataset_name
        )
    
    except Exception as e:
        logger.error(f"Ranking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")
