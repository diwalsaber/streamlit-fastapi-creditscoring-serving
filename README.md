# Credit Scoring Application

[![CI/CD Pipeline](https://github.com/your-org/credit-risk-api/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/credit-risk-api/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-ready credit scoring application for loan default prediction using LightGBM. Built with FastAPI (backend) and Streamlit (frontend), containerized with Docker, and featuring comprehensive testing, security, and monitoring.

> Based on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle competition dataset.

## ✨ Features

### 🚀 Production-Ready
- **Microservices Architecture**: Separate backend API and frontend UI services
- **Containerized Deployment**: Multi-stage Docker builds with non-root users
- **Health Monitoring**: Built-in health checks and readiness probes
- **Structured Logging**: Comprehensive logging with configurable levels
- **Error Handling**: Robust error handling with informative messages

### 🔒 Security
- **CORS Protection**: Configurable CORS middleware
- **Rate Limiting**: API rate limiting to prevent abuse
- **Input Validation**: Pydantic models for request/response validation
- **Security Headers**: X-Content-Type-Options, X-Frame-Options, CSP, etc.
- **Container Security**: Non-root users, minimal base images, vulnerability scanning

### 📊 ML & Explainability
- **LightGBM Model**: Fast and accurate gradient boosting model
- **SHAP Values**: Model explainability for predictions
- **Risk Categorization**: Automatic low/medium/high risk classification
- **Batch & Single Predictions**: Support for both new and existing clients

### 🧪 Testing & Quality
- **Comprehensive Tests**: Unit, integration, and API tests with pytest
- **Code Coverage**: >70% test coverage requirement
- **CI/CD Pipeline**: Automated testing, linting, and security scanning
- **Pre-commit Hooks**: Automatic code formatting and linting
- **Type Checking**: MyPy static type analysis

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Contributing](#contributing)

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose (recommended)
- Python 3.11+ (for local development)
- Git

### Running with Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/credit-risk-api.git
   cd credit-risk-api
   ```

2. **Download model and data files**
   ```bash
   # Place your model and data files in the appropriate directories
   # See DATA_README.md for details
   ./scripts/download_data.sh
   ```

3. **Build and run**
   ```bash
   docker-compose build
   docker-compose up
   ```

4. **Access the services**
   - **Streamlit UI**: http://localhost:8501
   - **FastAPI Docs**: http://localhost:8000/docs
   - **API Health Check**: http://localhost:8000/health

### Running Locally (Development)

1. **Set up backend**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python main.py
   ```

2. **Set up frontend** (in another terminal)
   ```bash
   cd frontend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   streamlit run ui.py
   ```

## 🏗️ Architecture

```
┌─────────────────┐         HTTP          ┌──────────────────┐
│                 │ ◄──────────────────── │                  │
│   Streamlit UI  │                       │   FastAPI        │
│   (Frontend)    │ ───────────────────► │   (Backend)      │
│   Port 8501     │      REST API         │   Port 8000      │
└─────────────────┘                       └──────────────────┘
                                                    │
                                                    │
                                          ┌─────────▼──────────┐
                                          │  LightGBM Model    │
                                          │  + Training Data   │
                                          └────────────────────┘
```

### Project Structure

```
.
├── backend/                 # FastAPI backend service
│   ├── api/                # API routes and middleware
│   ├── models/             # ML model wrapper
│   ├── schemas/            # Pydantic models
│   ├── utils/              # Utilities (logging, etc.)
│   ├── main.py             # Application entry point
│   ├── Dockerfile          # Multi-stage Docker build
│   └── requirements.txt    # Python dependencies
├── frontend/               # Streamlit frontend service
│   ├── components/         # UI components
│   ├── utils/              # Utilities (API client, etc.)
│   ├── ui.py               # Main UI application
│   ├── Dockerfile          # Multi-stage Docker build
│   └── requirements.txt    # Python dependencies
├── tests/                  # Test suite
│   ├── backend/            # Backend tests
│   ├── frontend/           # Frontend tests
│   └── integration/        # Integration tests
├── config/                 # Configuration module
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── .github/workflows/      # CI/CD pipelines
├── docker-compose.yml      # Multi-container orchestration
├── pyproject.toml          # Python project configuration
└── README.md              # This file
```

## 📡 API Documentation

### Endpoints

#### Health Check
```bash
GET /health
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "model_loaded": true,
  "data_loaded": true
}
```

#### Predict New Client
```bash
POST /api/v1/predict/new
```

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "probability": 0.23,
  "client_id": null,
  "risk_level": "low"
}
```

#### Predict Existing Client
```bash
POST /api/v1/predict/existing
```

**Request Body:**
```json
{
  "id_client": 100
}
```

**Response:**
```json
{
  "probability": 0.35,
  "client_id": 100,
  "risk_level": "medium"
}
```

For complete API documentation, visit `/docs` (Swagger UI) or `/redoc` when the backend is running.

## 🛠️ Development

### Setup Development Environment

1. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   pip install -r frontend/requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Code Quality Tools

- **Formatting**: `black backend/ frontend/`
- **Linting**: `ruff check backend/ frontend/`
- **Type Checking**: `mypy backend/`
- **Security Scan**: `bandit -r backend/`

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov=frontend --cov-report=html

# Run specific test file
pytest tests/backend/test_api.py

# Run with verbose output
pytest -v
```

### Test Coverage

Current coverage: **>70%**

View coverage report: `open htmlcov/index.html`

## 🚢 Deployment

### Docker Compose (Production)

```bash
# Build and run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Environment Variables

See `.env.example` for all available configuration options.

Key variables:
- `ENVIRONMENT`: `development`, `staging`, or `production`
- `LOG_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `BACKEND_URL`: Backend API URL for frontend
- `CORS_ORIGINS`: Allowed CORS origins
- `MODEL_PATH`: Path to model file
- `DATA_PATH`: Path to training data

### Heroku Deployment (with Dockhero)

1. Rename `docker-compose.yml` to `dockhero-compose.yml`
2. Create Heroku app: `heroku create <app-name>`
3. Install Dockhero: `heroku addons:create dockhero`
4. Deploy: `heroku dh:compose up -d --app <app-name>`
5. Get URL: `heroku dh:open --app <app-name>`
6. View logs: `heroku logs -p dockhero --app <app-name>`

## ⚙️ Configuration

### Backend Configuration

Edit `backend/main.py` or set environment variables:

```python
MODEL_PATH = os.getenv("MODEL_PATH", "model_saved.pkl")
DATA_PATH = os.getenv("DATA_PATH", "reduced_train.csv")
HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
PORT = int(os.getenv("BACKEND_PORT", "8000"))
```

### Frontend Configuration

Edit `frontend/ui.py` or set environment variables:

```python
BACKEND_URL = os.getenv("BACKEND_URL", "http://fastapi:8000")
```

## 📊 Monitoring

### Health Checks

Both services include health check endpoints:
- Backend: `http://localhost:8000/health`
- Frontend: `http://localhost:8501/_stcore/health`

### Logging

Structured logging is enabled by default. Logs include:
- Request/response details
- Prediction results
- Error stack traces
- Performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest`
5. Run linters: `pre-commit run --all-files`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle Competition
- FastAPI framework by Sebastián Ramírez
- Streamlit framework by Streamlit Inc.
- LightGBM by Microsoft

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using FastAPI, Streamlit, and LightGBM**
