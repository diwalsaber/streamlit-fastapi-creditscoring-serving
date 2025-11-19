# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-11-19

### 🎉 Major Refactoring - Production-Ready Release

Complete overhaul of the application to enterprise-grade quality standards.

### Added

#### Architecture
- Modular backend structure: `api/`, `models/`, `schemas/`, `utils/`
- API versioning with `/api/v1/*` endpoints
- Configuration management with `config/` module
- Environment-based configuration with `.env` support

#### Backend Features
- `ModelPredictor` class for ML model abstraction
- Structured logging with custom logger
- Comprehensive error handling with specific HTTP status codes
- Health check endpoints with model/data status
- Legacy endpoint proxies for backward compatibility
- Global exception handler

#### Security
- CORS middleware with configurable origins
- Rate limiting with `slowapi`
- Security headers (X-Frame-Options, CSP, HSTS, etc.)
- Input validation with Pydantic v2
- Non-root Docker containers
- Multi-stage Docker builds

#### Frontend
- `APIClient` utility class for backend communication
- Modern `@st.cache_data` decorator (replaced deprecated `@st.cache`)
- Improved error handling with user-friendly messages
- Risk level display (low/medium/high)
- Better UX with success/error messages

#### Testing & Quality
- Pytest test suite with fixtures
- Test configuration (`pytest.ini`, `pyproject.toml`)
- Pre-commit hooks configuration
- GitHub Actions CI/CD pipeline
- Code quality tools (black, ruff, mypy, bandit)

#### Documentation
- Comprehensive README with badges and guides
- API documentation with examples
- Architecture diagrams
- DATA_README for large file management
- CHANGELOG (this file)
- CONTRIBUTING guide
- Deployment instructions

#### Infrastructure
- Optimized Dockerfiles with multi-stage builds
- Updated docker-compose.yml (v3.8) with health checks
- `.dockerignore` files for both services
- Development requirements file

### Changed

#### Breaking Changes
- **API endpoints moved to `/api/v1/*`**
  - Old: `/predict_new` → New: `/api/v1/predict/new`
  - Old: `/predict_previous` → New: `/api/v1/predict/existing`
  - Legacy endpoints still work but log deprecation warnings

- **Docker containers now run as non-root user (`appuser:1000`)**

- **Response format changed**
  - Now returns: `{"probability": float, "risk_level": str, "client_id": int|null}`
  - Old format: Just the probability value

#### Performance Improvements
- **CRITICAL FIX**: Training data now loaded once at startup (was 47MB per request!)
- Optimized Docker images (~40% smaller)
- Improved caching strategies

#### Dependencies
- Updated all dependencies to latest secure versions
- `fastapi`: 0.88.0 → 0.115.0
- `streamlit`: 1.15.1 → 1.40.1
- `pandas`: 1.5.1 → 2.2.3
- `numpy`: 1.23.2 → 2.1.3
- `lightgbm`: 3.3.x → 4.5.0
- `pillow`: 9.0.1 → 11.0.0 (security fix)
- `pydantic`: 1.x → 2.9.2
- Added `pydantic-settings`, `slowapi`, `prometheus-client`

#### Configuration
- Environment variables now use CSV format for lists (not JSON)
- Simplified data paths (no `/app/data`, `/app/models`)
- Docker volumes mount entire service directories

### Fixed

- API contract mismatch between frontend and backend
- Frontend not using APIClient module
- Docker volume paths pointing to non-existent directories
- CORS configuration format inconsistency
- Deprecated `@st.cache` decorator
- Legacy endpoints returning messages instead of predictions
- CSV file loaded on every request (performance bug)
- No error handling in predictor
- Hardcoded file paths
- Using `print()` for debugging instead of logging
- Root containers (security issue)
- Missing input validation
- Outdated dependencies with known CVEs

### Removed

- Commented-out code blocks
- Unused dependencies (`pickle5`, `ipython`)
- Duplicate dependencies in requirements.txt
- Development flags from production Docker images
- Unused Docker volumes in compose file

### Security

- Fixed all known CVE vulnerabilities in dependencies
- Added comprehensive security middleware
- Implemented input validation for all endpoints
- Removed root user from containers
- Added security headers to all responses

## [1.0.0] - Previous Version

Initial proof-of-concept version with:
- Basic FastAPI backend
- Streamlit frontend
- Docker deployment
- LightGBM model integration
- SHAP explanations

---

## Migration Guide

### From 1.0.0 to 2.0.0

#### For API Clients

Update your API calls to use the new versioned endpoints:

```python
# Old
response = requests.post("http://backend:8000/predict_new", json=data)

# New (recommended)
response = requests.post("http://backend:8000/api/v1/predict/new", json=data)

# Legacy endpoints still work but will be removed in 3.0.0
response = requests.post("http://backend:8000/predict_new", json=data)
```

#### For Docker Deployments

1. Update docker-compose.yml to version 3.8
2. Ensure data files are in `backend/` and `frontend/` directories
3. Update CORS_ORIGINS to CSV format in environment

#### For Developers

1. Install new development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

3. Run tests:
   ```bash
   pytest
   ```

---

[2.0.0]: https://github.com/your-org/credit-risk-api/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/your-org/credit-risk-api/releases/tag/v1.0.0
