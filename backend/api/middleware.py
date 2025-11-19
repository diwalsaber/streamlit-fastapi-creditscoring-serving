"""Security middleware for the API."""
import os
from fastapi import Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Callable


# Rate limiter
limiter = Limiter(key_func=get_remote_address)


def get_cors_middleware_config() -> dict:
    """
    Get CORS middleware configuration from environment.

    Returns:
        Dictionary with CORS configuration
    """
    # Default origins for development
    default_origins = [
        "http://localhost:8501",
        "http://localhost:3000",
        "http://127.0.0.1:8501",
    ]

    # Get origins from environment (comma-separated string)
    origins_str = os.getenv("CORS_ORIGINS", "")
    if origins_str:
        origins = [o.strip() for o in origins_str.split(",")]
    else:
        origins = default_origins

    return {
        "allow_origins": origins,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["*"],
    }


async def security_headers_middleware(request: Request, call_next: Callable):
    """
    Add security headers to all responses.

    Args:
        request: The incoming request
        call_next: The next middleware/endpoint to call

    Returns:
        Response with security headers
    """
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    return response


def configure_security(app):
    """
    Configure all security middleware for the application.

    Args:
        app: FastAPI application instance
    """
    # CORS
    cors_config = get_cors_middleware_config()
    app.add_middleware(CORSMiddleware, **cors_config)

    # Security headers
    app.middleware("http")(security_headers_middleware)

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
