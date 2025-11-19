"""Tests for model predictor."""
import pytest
from unittest.mock import Mock, patch, mock_open
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.models.predictor import ModelPredictor


class TestModelPredictor:
    """Tests for ModelPredictor class."""

    @pytest.fixture
    def mock_model(self):
        """Mock ML model."""
        model = Mock()
        model.predict_proba = Mock(return_value=[[0.7, 0.3]])
        return model

    @pytest.fixture
    def mock_data(self):
        """Mock training data."""
        return pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })

    def test_get_risk_level_low(self):
        """Test risk level categorization - low."""
        predictor = ModelPredictor.__new__(ModelPredictor)
        assert predictor.get_risk_level(0.2) == "low"

    def test_get_risk_level_medium(self):
        """Test risk level categorization - medium."""
        predictor = ModelPredictor.__new__(ModelPredictor)
        assert predictor.get_risk_level(0.45) == "medium"

    def test_get_risk_level_high(self):
        """Test risk level categorization - high."""
        predictor = ModelPredictor.__new__(ModelPredictor)
        assert predictor.get_risk_level(0.75) == "high"

    def test_model_not_found(self):
        """Test error when model file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ModelPredictor(model_path="nonexistent.pkl")

    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    def test_predict_new_client_no_model(self, mock_pickle, mock_file):
        """Test prediction fails when model not loaded."""
        mock_pickle.return_value = None

        with patch('pathlib.Path.exists', return_value=True):
            predictor = ModelPredictor(model_path="test.pkl")
            predictor.model = None

            with pytest.raises(ValueError, match="Model not loaded"):
                predictor.predict_new_client({})

    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    @patch('pathlib.Path.exists')
    def test_predict_existing_client_invalid_id(self, mock_exists, mock_pickle, mock_file):
        """Test prediction with invalid client ID."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_pickle.return_value = mock_model

        predictor = ModelPredictor(model_path="test.pkl", data_path="test.csv")
        predictor.training_data = pd.DataFrame({'col1': [1, 2, 3]})

        with pytest.raises(ValueError, match="Invalid client_id"):
            predictor.predict_existing_client(999)
