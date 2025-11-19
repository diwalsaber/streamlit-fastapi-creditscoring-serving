"""API route handlers."""
from fastapi import APIRouter, HTTPException, status
from backend.schemas.prediction import (
    ClientInput,
    ClientID,
    PredictionResponse,
    HealthResponse
)
from backend.models.predictor import ModelPredictor
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

# Router for API endpoints
router = APIRouter()

# Global predictor instance (will be set in main.py)
predictor: ModelPredictor = None


def set_predictor(model_predictor: ModelPredictor) -> None:
    """Set the global predictor instance."""
    global predictor
    predictor = model_predictor


@router.get("/", response_model=HealthResponse, tags=["health"])
@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Health status including model and data loading status
    """
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        model_loaded=predictor.is_loaded if predictor else False,
        data_loaded=predictor.is_data_loaded if predictor else False
    )


@router.post(
    "/api/v1/predict/new",
    response_model=PredictionResponse,
    tags=["predictions"],
    summary="Predict for new client",
    description="Predict loan default probability for a new client with provided features"
)
async def predict_new_client(client_data: ClientInput) -> PredictionResponse:
    """
    Predict probability of loan default for a new client.

    Args:
        client_data: Client features

    Returns:
        Prediction response with probability and risk level

    Raises:
        HTTPException: If prediction fails
    """
    if not predictor or not predictor.is_loaded:
        logger.error("Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Convert Pydantic model to dict
        client_dict = client_data.model_dump()

        # Make prediction
        probability = predictor.predict_new_client(client_dict)

        # Get risk level
        risk_level = predictor.get_risk_level(probability)

        logger.info(f"New client prediction: {probability:.4f} ({risk_level} risk)")

        return PredictionResponse(
            probability=probability,
            client_id=None,
            risk_level=risk_level
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/api/v1/predict/existing",
    response_model=PredictionResponse,
    tags=["predictions"],
    summary="Predict for existing client",
    description="Predict loan default probability for an existing client by ID"
)
async def predict_existing_client(client_id_input: ClientID) -> PredictionResponse:
    """
    Predict probability of loan default for an existing client.

    Args:
        client_id_input: Client ID (row index in training data)

    Returns:
        Prediction response with probability and risk level

    Raises:
        HTTPException: If client ID is invalid or prediction fails
    """
    if not predictor or not predictor.is_loaded:
        logger.error("Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    if not predictor.is_data_loaded:
        logger.error("Training data not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Training data not available for existing client predictions"
        )

    try:
        client_id = client_id_input.id_client

        # Make prediction
        probability = predictor.predict_existing_client(client_id)

        # Get risk level
        risk_level = predictor.get_risk_level(probability)

        logger.info(
            f"Existing client {client_id} prediction: {probability:.4f} ({risk_level} risk)"
        )

        return PredictionResponse(
            probability=probability,
            client_id=client_id,
            risk_level=risk_level
        )

    except ValueError as e:
        logger.warning(f"Invalid client ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get(
    "/api/v1/clients/count",
    tags=["data"],
    summary="Get client count",
    description="Get the number of clients in the training dataset"
)
async def get_client_count() -> dict:
    """
    Get the number of clients in training data.

    Returns:
        Dictionary with client count
    """
    if not predictor or not predictor.is_data_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Training data not loaded"
        )

    return {
        "count": predictor.data_size,
        "valid_range": f"0 to {predictor.data_size - 1}"
    }
