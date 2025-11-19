"""Application settings and configuration management."""
import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Backend settings
    backend_host: str = Field(default="0.0.0.0", env="BACKEND_HOST")
    backend_port: int = Field(default=8000, env="BACKEND_PORT")
    backend_workers: int = Field(default=4, env="BACKEND_WORKERS")
    backend_reload: bool = Field(default=False, env="BACKEND_RELOAD")

    # Frontend settings
    frontend_port: int = Field(default=8501, env="FRONTEND_PORT")
    backend_url: str = Field(default="http://fastapi:8000", env="BACKEND_URL")

    # Application settings
    environment: str = Field(default="production", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    debug: bool = Field(default=False, env="DEBUG")

    # Security
    cors_origins: List[str] = Field(default=["http://localhost:8501"], env="CORS_ORIGINS")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    api_key_required: bool = Field(default=False, env="API_KEY_REQUIRED")

    # Data paths
    model_path: str = Field(default="models/model_saved.pkl", env="MODEL_PATH")
    data_path: str = Field(default="data/reduced_train.csv", env="DATA_PATH")
    explainer_path: str = Field(default="models/explainer.pkl", env="EXPLAINER_PATH")
    shap_values_path: str = Field(default="models/shap_values.pkl", env="SHAP_VALUES_PATH")

    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid option."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is a valid option."""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
