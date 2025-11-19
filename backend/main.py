"""
Credit Scoring API - FastAPI Backend

A production-ready API for credit default prediction using LightGBM.
Features:
- Prediction for new and existing clients
- Comprehensive error handling
- Request validation
- Health monitoring
- Structured logging
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models.predictor import ModelPredictor
from backend.api.routes import router, set_predictor
from backend.api.middleware import configure_security
from backend.utils.logger import setup_logger
from backend.schemas.prediction import ClientInput, ClientID

# Setup logging
logger = setup_logger(__name__)

# Configuration from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "model_saved.pkl")
DATA_PATH = os.getenv("DATA_PATH", "reduced_train.csv")
HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
PORT = int(os.getenv("BACKEND_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RELOAD = os.getenv("BACKEND_RELOAD", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Credit Scoring API...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'production')}")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Data path: {DATA_PATH}")

    try:
        # Initialize predictor with model and data
        predictor = ModelPredictor(
            model_path=MODEL_PATH,
            data_path=DATA_PATH
        )

        # Set the predictor in routes
        set_predictor(predictor)

        logger.info("✓ Model and data loaded successfully")
        logger.info(f"✓ Training data: {predictor.data_size} clients")
        logger.info("✓ API ready to accept requests")

    except FileNotFoundError as e:
        logger.error(f"✗ File not found: {e}")
        logger.error("API will start but predictions will fail")
        logger.error("Please ensure model and data files are available")
    except Exception as e:
        logger.error(f"✗ Startup error: {e}")
        logger.error("API will start but may not function correctly")

    yield

    # Shutdown
    logger.info("Shutting down Credit Scoring API...")


# Create FastAPI application
app = FastAPI(
    title="Credit Scoring API",
    description=(
        "Production-ready API for loan default prediction using LightGBM. "
        "Provides predictions for both new and existing clients with "
        "comprehensive error handling and monitoring."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure security (CORS, rate limiting, security headers)
configure_security(app)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__
        }
    )


# Include routers
app.include_router(router)


# Legacy endpoint proxies for backward compatibility
@app.get("/home")
async def legacy_home():
    """Legacy health check endpoint - proxies to /health."""
    from backend.api.routes import health_check
    logger.warning("Legacy /home endpoint used - please migrate to /health")
    return await health_check()


@app.post("/predict_new")
async def legacy_predict_new(client_data: ClientInput):
    """Legacy prediction endpoint - proxies to /api/v1/predict/new."""
    from backend.api.routes import predict_new_client
    logger.warning("Legacy /predict_new endpoint used - please migrate to /api/v1/predict/new")
    response = await predict_new_client(client_data)
    # Return just the probability for backward compatibility
    return response.probability


@app.post("/predict_previous")
async def legacy_predict_previous(client_id_input: ClientID):
    """Legacy prediction endpoint - proxies to /api/v1/predict/existing."""
    from backend.api.routes import predict_existing_client
    logger.warning("Legacy /predict_previous endpoint used - please migrate to /api/v1/predict/existing")
    response = await predict_existing_client(client_id_input)
    # Return just the probability for backward compatibility
    return response.probability


if __name__ == "__main__":
    """Run the application with uvicorn."""
    logger.info(f"Starting server on {HOST}:{PORT}")
    logger.info(f"Reload mode: {RELOAD}")
    logger.info(f"Log level: {LOG_LEVEL}")

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        log_level=LOG_LEVEL.lower(),
        access_log=True
    )
