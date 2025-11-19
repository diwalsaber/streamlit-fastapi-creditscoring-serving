"""Pydantic models for prediction requests and responses."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ClientInput(BaseModel):
    """Input data for new client prediction."""

    EXT_SOURCE_1: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="External source score 1"
    )
    EXT_SOURCE_2: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="External source score 2"
    )
    EXT_SOURCE_3: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="External source score 3"
    )
    DAYS_BIRTH: float = Field(
        ...,
        description="Client's age in days (negative number)"
    )
    AMT_GOODS_PRICE: float = Field(
        ...,
        ge=0.0,
        description="Price of goods for the loan"
    )
    AMT_CREDIT: float = Field(
        ...,
        ge=0.0,
        description="Credit amount of the loan"
    )
    AMT_ANNUITY: float = Field(
        ...,
        ge=0.0,
        description="Loan annuity"
    )
    DAYS_EMPLOYED: float = Field(
        ...,
        description="How many days before the application the person started current employment"
    )
    CODE_GENDER: float = Field(
        ...,
        description="Gender of the client (0 or 1)"
    )
    AMT_INCOME_TOTAL: float = Field(
        ...,
        ge=0.0,
        description="Income of the client"
    )
    DAYS_EMPLOYED_PERC: float = Field(
        ...,
        description="Days employed percentage"
    )
    INCOME_CREDIT_PERC: float = Field(
        ...,
        description="Income to credit percentage"
    )
    ANNUITY_INCOME_PERC: float = Field(
        ...,
        description="Annuity to income percentage"
    )
    PAYMENT_RATE: float = Field(
        ...,
        description="Payment rate"
    )

    @field_validator("CODE_GENDER")
    @classmethod
    def validate_gender(cls, v: float) -> float:
        """Validate gender is 0 or 1."""
        if v not in [0.0, 1.0]:
            raise ValueError("CODE_GENDER must be 0 or 1")
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "EXT_SOURCE_1": 0.5,
                "EXT_SOURCE_2": 0.6,
                "EXT_SOURCE_3": 0.4,
                "DAYS_BIRTH": -15000,
                "AMT_GOODS_PRICE": 450000.0,
                "AMT_CREDIT": 500000.0,
                "AMT_ANNUITY": 25000.0,
                "DAYS_EMPLOYED": -2000,
                "CODE_GENDER": 1.0,
                "AMT_INCOME_TOTAL": 150000.0,
                "DAYS_EMPLOYED_PERC": 0.13,
                "INCOME_CREDIT_PERC": 0.30,
                "ANNUITY_INCOME_PERC": 0.17,
                "PAYMENT_RATE": 0.05
            }
        }


class ClientID(BaseModel):
    """Client ID for existing client lookup."""

    id_client: int = Field(
        ...,
        ge=0,
        description="Client ID (row index in training data)"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "id_client": 100
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of loan default"
    )
    client_id: Optional[int] = Field(
        None,
        description="Client ID (for existing clients)"
    )
    risk_level: str = Field(
        ...,
        description="Risk level category"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "probability": 0.23,
                "client_id": None,
                "risk_level": "low"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    data_loaded: bool = Field(..., description="Whether training data is loaded")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True,
                "data_loaded": True
            }
        }
