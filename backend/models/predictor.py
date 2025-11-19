"""ML model predictor with caching and error handling."""
import pickle
from pathlib import Path
from typing import Optional, Any
import pandas as pd
import numpy as np
from functools import lru_cache

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelPredictor:
    """Wrapper for ML model with prediction capabilities."""

    def __init__(self, model_path: str, data_path: Optional[str] = None):
        """
        Initialize the predictor with model and optional training data.

        Args:
            model_path: Path to the pickled model file
            data_path: Optional path to training data CSV

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path) if data_path else None
        self.model: Optional[Any] = None
        self.training_data: Optional[pd.DataFrame] = None

        self._load_model()
        if self.data_path:
            self._load_training_data()

    def _load_model(self) -> None:
        """Load the ML model from pickle file."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_training_data(self) -> None:
        """Load training data for existing client lookups."""
        if not self.data_path or not self.data_path.exists():
            logger.warning(f"Training data file not found: {self.data_path}")
            return

        try:
            self.training_data = pd.read_csv(self.data_path)
            logger.info(
                f"Training data loaded: {len(self.training_data)} rows, "
                f"{len(self.training_data.columns)} columns"
            )
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            # Don't raise - this is optional

    def predict_new_client(self, client_data: dict) -> float:
        """
        Predict probability for a new client.

        Args:
            client_data: Dictionary with client features

        Returns:
            Predicted probability of default

        Raises:
            ValueError: If model is not loaded
            Exception: If prediction fails
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            # Convert to DataFrame with correct column order
            df = pd.DataFrame([client_data])

            # Predict probability (class 1 = default)
            probability = float(self.model.predict_proba(df)[0][1])

            logger.info(f"Prediction for new client: {probability:.4f}")
            return probability

        except Exception as e:
            logger.error(f"Prediction failed for new client: {e}")
            raise

    def predict_existing_client(self, client_id: int) -> float:
        """
        Predict probability for an existing client by ID.

        Args:
            client_id: Row index in training data

        Returns:
            Predicted probability of default

        Raises:
            ValueError: If training data not loaded or invalid ID
            Exception: If prediction fails
        """
        if self.training_data is None:
            raise ValueError("Training data not loaded")

        if client_id < 0 or client_id >= len(self.training_data):
            raise ValueError(
                f"Invalid client_id: {client_id}. "
                f"Must be between 0 and {len(self.training_data) - 1}"
            )

        try:
            # Get client data by index
            client_data = self.training_data.iloc[[client_id]]

            # Predict probability
            probability = float(self.model.predict_proba(client_data)[0][1])

            logger.info(
                f"Prediction for existing client {client_id}: {probability:.4f}"
            )
            return probability

        except Exception as e:
            logger.error(f"Prediction failed for client {client_id}: {e}")
            raise

    def get_risk_level(self, probability: float) -> str:
        """
        Categorize risk level based on probability.

        Args:
            probability: Predicted probability of default

        Returns:
            Risk level category: 'low', 'medium', or 'high'
        """
        if probability < 0.3:
            return "low"
        elif probability < 0.6:
            return "medium"
        else:
            return "high"

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    @property
    def is_data_loaded(self) -> bool:
        """Check if training data is loaded."""
        return self.training_data is not None

    @property
    def data_size(self) -> int:
        """Get number of clients in training data."""
        return len(self.training_data) if self.training_data is not None else 0
