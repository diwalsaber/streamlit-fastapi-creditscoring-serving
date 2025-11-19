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


# Legacy endpoint redirects for backward compatibility
@app.get("/home")
async def legacy_home():
    """Redirect to health endpoint."""
    return {"message": "Use /health endpoint for health checks"}


@app.post("/predict_new")
async def legacy_predict_new():
    """Redirect to new endpoint."""
    return {
        "message": "This endpoint is deprecated. Use POST /api/v1/predict/new instead"
    }


@app.post("/predict_previous")
async def legacy_predict_previous():
    """Redirect to new endpoint."""
    return {
        "message": "This endpoint is deprecated. Use POST /api/v1/predict/existing instead"
    }


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
