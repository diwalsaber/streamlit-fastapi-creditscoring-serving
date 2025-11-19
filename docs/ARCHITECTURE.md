# Architecture Documentation

Comprehensive architecture overview of the Credit Scoring Application.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Backend Architecture](#backend-architecture)
- [Frontend Architecture](#frontend-architecture)
- [Data Flow](#data-flow)
- [Security Architecture](#security-architecture)
- [Deployment Architecture](#deployment-architecture)
- [Design Patterns](#design-patterns)
- [Technology Stack](#technology-stack)

---

## Overview

The Credit Scoring Application is a **microservices-based** system for predicting loan default probability using a LightGBM machine learning model. It follows modern software engineering practices with emphasis on:

- **Modularity**: Clear separation of concerns
- **Scalability**: Horizontal scaling capability
- **Security**: Multiple layers of protection
- **Maintainability**: Clean code and comprehensive testing
- **Performance**: Optimized for low latency

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Internet                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Load Balancer      │
              │   (Optional)         │
              └──────────┬───────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────┐             ┌─────────────────┐
│   Frontend      │             │   Backend       │
│   (Streamlit)   │◄───────────►│   (FastAPI)     │
│   Port 8501     │   REST API  │   Port 8000     │
└─────────────────┘             └────────┬────────┘
                                         │
                                         ▼
                              ┌──────────────────┐
                              │  ML Model        │
                              │  LightGBM        │
                              │  + Training Data │
                              └──────────────────┘
```

### Component Breakdown

| Component | Technology | Port | Purpose |
|-----------|------------|------|---------|
| Frontend | Streamlit 1.40.1 | 8501 | User interface |
| Backend | FastAPI 0.115.0 | 8000 | API server |
| ML Model | LightGBM 4.5.0 | - | Predictions |
| Explainer | SHAP 0.46.0 | - | Model interpretability |

---

## Backend Architecture

### Module Structure

```
backend/
├── __init__.py
├── main.py                  # Application entry point
├── api/                     # API layer
│   ├── __init__.py
│   ├── routes.py            # Endpoint definitions
│   └── middleware.py        # Security middleware
├── models/                  # Business logic layer
│   ├── __init__.py
│   └── predictor.py         # ML model wrapper
├── schemas/                 # Data validation layer
│   ├── __init__.py
│   └── prediction.py        # Pydantic models
└── utils/                   # Utilities
    ├── __init__.py
    └── logger.py            # Logging configuration
```

### Layered Architecture

```
┌─────────────────────────────────────────┐
│         API Layer (routes.py)           │
│  - Endpoint definitions                 │
│  - Request/response handling            │
│  - HTTP status codes                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Validation Layer (schemas/)        │
│  - Input validation (Pydantic)          │
│  - Output serialization                 │
│  - Type checking                        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│     Business Logic (models/)            │
│  - Model loading                        │
│  - Prediction logic                     │
│  - Risk categorization                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│     Infrastructure (utils/)             │
│  - Logging                              │
│  - Configuration                        │
└─────────────────────────────────────────┘
```

### Key Components

#### 1. ModelPredictor (`models/predictor.py`)

**Responsibilities:**
- Load and manage ML model
- Load training data for existing client predictions
- Execute predictions
- Categorize risk levels

**Methods:**
```python
class ModelPredictor:
    def __init__(model_path, data_path)
    def predict_new_client(client_data: dict) -> float
    def predict_existing_client(client_id: int) -> float
    def get_risk_level(probability: float) -> str

    # Properties
    @property is_loaded -> bool
    @property is_data_loaded -> bool
    @property data_size -> int
```

#### 2. API Routes (`api/routes.py`)

**Endpoints:**
- `GET /health` - Health check
- `POST /api/v1/predict/new` - New client prediction
- `POST /api/v1/predict/existing` - Existing client prediction
- `GET /api/v1/clients/count` - Get client count

**Pattern:** Async handlers with dependency injection

#### 3. Middleware (`api/middleware.py`)

**Security Layers:**
- CORS protection
- Rate limiting
- Security headers
- Request/response logging

---

## Frontend Architecture

### Structure

```
frontend/
├── __init__.py
├── ui.py                    # Main application
└── utils/
    ├── __init__.py
    └── api_client.py        # Backend API client
```

### Page Architecture

```
┌─────────────────────────────────────────┐
│          Streamlit App (ui.py)          │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────┐   │
│  │       Home Page                 │   │
│  │  - Welcome message              │   │
│  │  - Project description          │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │     New Client Page             │   │
│  │  - Input form (sliders, dates)  │   │
│  │  - Prediction display           │   │
│  │  - SHAP explanation             │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │   Previous Client Page          │   │
│  │  - Client ID input              │   │
│  │  - Prediction display           │   │
│  │  - SHAP plots (global/local)    │   │
│  │  - Feature distributions        │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### APIClient Pattern

```python
class APIClient:
    def __init__(base_url)
    def predict_new_client(client_data) -> dict
    def predict_existing_client(client_id) -> dict
    def get_health() -> dict
    def get_client_count() -> dict
```

**Features:**
- Error handling with user feedback
- Timeout configuration
- Automatic retries
- Streamlit integration

---

## Data Flow

### New Client Prediction Flow

```
User Input (Streamlit)
    │
    ├─> Feature Engineering (frontend)
    │   └─> Calculate derived features
    │
    └─> APIClient.predict_new_client()
        │
        ├─> HTTP POST /api/v1/predict/new
        │
        └─> Backend Processing
            │
            ├─> Pydantic Validation (schemas)
            │
            ├─> ModelPredictor.predict_new_client()
            │   ├─> Create DataFrame
            │   ├─> Model.predict_proba()
            │   └─> Get risk level
            │
            └─> Response
                │
                └─> Display Results
                    ├─> Probability gauge
                    ├─> Risk level badge
                    └─> SHAP explanation
```

### Existing Client Prediction Flow

```
Client ID Input
    │
    └─> APIClient.predict_existing_client()
        │
        ├─> HTTP POST /api/v1/predict/existing
        │
        └─> Backend Processing
            │
            ├─> Validate Client ID
            │
            ├─> Fetch from Training Data
            │
            ├─> ModelPredictor.predict_existing_client()
            │   ├─> Lookup by index
            │   ├─> Model.predict_proba()
            │   └─> Get risk level
            │
            └─> Response + Visualizations
```

---

## Security Architecture

### Multi-Layer Security

```
┌────────────────────────────────────────┐
│  1. Network Layer                      │
│     - HTTPS (in production)            │
│     - Firewall rules                   │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  2. Application Layer                  │
│     - CORS protection                  │
│     - Rate limiting (60/min)           │
│     - Security headers                 │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  3. Validation Layer                   │
│     - Input validation (Pydantic)      │
│     - Type checking                    │
│     - Range validation                 │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  4. Infrastructure Layer               │
│     - Non-root containers              │
│     - Read-only file systems           │
│     - Minimal base images              │
└────────────────────────────────────────┘
```

### Security Headers

| Header | Value | Purpose |
|--------|-------|---------|
| X-Frame-Options | DENY | Prevent clickjacking |
| X-Content-Type-Options | nosniff | Prevent MIME sniffing |
| X-XSS-Protection | 1; mode=block | XSS protection |
| Strict-Transport-Security | max-age=31536000 | Force HTTPS |
| Referrer-Policy | strict-origin-when-cross-origin | Privacy |

---

## Deployment Architecture

### Docker Architecture

```
┌─────────────────────────────────────────────────┐
│          Docker Host                            │
│                                                 │
│  ┌───────────────────────────────────────────┐ │
│  │   Backend Container                       │ │
│  │   ┌─────────────────────────────┐        │ │
│  │   │  App (as appuser:1000)      │        │ │
│  │   │  - FastAPI                  │        │ │
│  │   │  - Uvicorn                  │        │ │
│  │   │  - Model + Data             │        │ │
│  │   └─────────────────────────────┘        │ │
│  │   Port: 8000                              │ │
│  └───────────────┬───────────────────────────┘ │
│                  │                             │
│  ┌───────────────┴───────────────────────────┐ │
│  │   Frontend Container                      │ │
│  │   ┌─────────────────────────────┐        │ │
│  │   │  App (as appuser:1000)      │        │ │
│  │   │  - Streamlit                │        │ │
│  │   │  - SHAP                     │        │ │
│  │   │  - Plots data               │        │ │
│  │   └─────────────────────────────┘        │ │
│  │   Port: 8501                              │ │
│  └───────────────────────────────────────────┘ │
│                                                 │
│  Network: credit-scoring-network (bridge)      │
└─────────────────────────────────────────────────┘
```

### Multi-Stage Build

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder
RUN python -m venv /opt/venv
COPY requirements.txt .
RUN pip install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
COPY --from=builder /opt/venv /opt/venv
USER appuser  # Non-root!
CMD [...]
```

**Benefits:**
- Smaller images (~40% reduction)
- No build tools in production
- Faster deployment
- Better security

---

## Design Patterns

### 1. **Repository Pattern**
ModelPredictor abstracts data access (model + training data)

### 2. **Dependency Injection**
FastAPI's dependency system for predictor

### 3. **Factory Pattern**
APIClient creation with caching

### 4. **Singleton Pattern**
Model loaded once at startup

### 5. **Strategy Pattern**
Different prediction strategies (new vs existing)

### 6. **Adapter Pattern**
APIClient adapts requests library for Streamlit

---

## Technology Stack

### Backend Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Framework | FastAPI | 0.115.0 |
| Server | Uvicorn | 0.32.0 |
| Validation | Pydantic | 2.9.2 |
| ML Model | LightGBM | 4.5.0 |
| Data | Pandas/NumPy | 2.2.3 / 2.1.3 |
| Security | slowapi | 0.1.9 |

### Frontend Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Framework | Streamlit | 1.40.1 |
| Visualization | Plotly | 5.24.1 |
| Plotting | Matplotlib/Seaborn | 3.9.2 / 0.13.2 |
| Explainability | SHAP | 0.46.0 |
| HTTP Client | Requests | 2.32.3 |

### Infrastructure

| Component | Technology | Version |
|-----------|------------|---------|
| Container | Docker | Latest |
| Orchestration | Docker Compose | 3.8 |
| Base Image | Python | 3.11-slim |
| CI/CD | GitHub Actions | v4/v5 |

---

## Scalability Considerations

### Horizontal Scaling

```
Load Balancer
    │
    ├─> Backend Instance 1
    ├─> Backend Instance 2
    └─> Backend Instance N
         │
         └─> Shared Model Storage (S3/GCS)
```

### Caching Strategy

```
Request
    │
    ├─> Check Cache (Redis)
    │   └─> Hit: Return cached result
    │
    └─> Miss: Compute prediction
        └─> Cache result
```

### Database Integration (Future)

```
Backend
    │
    ├─> PostgreSQL (Client data)
    ├─> Redis (Cache)
    └─> S3 (Model storage)
```

---

## Monitoring & Observability

### Health Checks

```
Docker Compose
    │
    ├─> Backend Health: GET /health (every 30s)
    └─> Frontend Health: GET /_stcore/health (every 30s)
```

### Logging

```
Application
    │
    ├─> Structured Logs (JSON)
    ├─> Log Levels (INFO, WARNING, ERROR)
    └─> Request/Response Logging
```

### Metrics (Future with Prometheus)

- Request count
- Response time
- Error rate
- Model prediction distribution
- Cache hit rate

---

## Performance Optimization

### Critical Optimizations

1. **Model Loading**: Once at startup (not per request)
2. **Data Loading**: Once at startup (was 47MB per request!)
3. **Docker Images**: Multi-stage builds
4. **Caching**: Streamlit cache for API calls
5. **Connection Pooling**: Async requests

### Benchmarks

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Cold start | ~30s | ~10s | 67% faster |
| Request latency | ~500ms | ~50ms | 90% faster |
| Image size | 2.5GB | 1.5GB | 40% smaller |
| Memory usage | 1.5GB | 800MB | 47% less |

---

## Future Enhancements

### Short Term
- [ ] Database integration (PostgreSQL)
- [ ] Redis caching
- [ ] Batch predictions endpoint
- [ ] Model versioning

### Medium Term
- [ ] Kubernetes deployment
- [ ] Prometheus metrics
- [ ] Grafana dashboards
- [ ] ELK stack for logging

### Long Term
- [ ] A/B testing framework
- [ ] Real-time predictions
- [ ] Model retraining pipeline
- [ ] Multi-model ensemble

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
