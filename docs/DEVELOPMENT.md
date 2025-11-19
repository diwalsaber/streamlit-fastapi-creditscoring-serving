# Development Guide

Complete guide for developers working on the Credit Scoring Application.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Running the Application](#running-the-application)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Debugging](#debugging)
- [Code Quality](#code-quality)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Python**: 3.11 or higher
- **Docker**: Latest version
- **Docker Compose**: Latest version
- **Git**: Latest version
- **IDE**: VS Code, PyCharm, or similar

### Recommended Tools

- **pyenv**: For Python version management
- **virtualenv**: For isolated Python environments
- **httpie** or **curl**: For API testing
- **Postman**: For API exploration

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/credit-scoring.git
cd credit-scoring
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
# Install production dependencies
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 4. Set Up Pre-commit Hooks

```bash
pre-commit install
```

### 5. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# For local development, defaults should work
```

### 6. Obtain Data Files

See `DATA_README.md` for instructions on obtaining model and data files.
Place files in appropriate directories:
- Backend: `backend/model_saved.pkl`, `backend/reduced_train.csv`
- Frontend: `frontend/explainer.pkl`, `frontend/shap_values.pkl`, `frontend/result_for_plot.csv`

---

## Development Environment

### VS Code Setup

#### Recommended Extensions

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "charliermarsh.ruff",
    "ms-python.black-formatter",
    "tamasfe.even-better-toml",
    "redhat.vscode-yaml",
    "ms-azuretools.vscode-docker",
    "eamodio.gitlens"
  ]
}
```

#### Settings

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

### PyCharm Setup

1. **Open Project**: File → Open → Select project directory
2. **Set Interpreter**: Settings → Project → Python Interpreter → Add → Virtualenv
3. **Enable Tools**:
   - Tools → Black
   - Tools → External Tools → Ruff
4. **Configure Tests**: Run → Edit Configurations → Add pytest

---

## Running the Application

### Option 1: Docker (Recommended)

```bash
# Build images
docker-compose build

# Start services
docker-compose up

# Or run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Access:**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000/docs

### Option 2: Local Development

#### Terminal 1 - Backend

```bash
cd backend
source ../venv/bin/activate

# Set environment variables
export MODEL_PATH=model_saved.pkl
export DATA_PATH=reduced_train.csv
export LOG_LEVEL=DEBUG

# Run with auto-reload
python main.py
```

#### Terminal 2 - Frontend

```bash
cd frontend
source ../venv/bin/activate

# Set environment variables
export BACKEND_URL=http://localhost:8000

# Run Streamlit
streamlit run ui.py
```

### Option 3: Development with Docker Override

Create `docker-compose.override.yml`:

```yaml
version: '3.8'

services:
  backend:
    volumes:
      - ./backend:/app:rw  # Read-write for development
    environment:
      - BACKEND_RELOAD=true
      - LOG_LEVEL=DEBUG

  frontend:
    volumes:
      - ./frontend:/app:rw
```

Then run:
```bash
docker-compose up
```

---

## Development Workflow

### Daily Workflow

1. **Pull Latest Changes**
   ```bash
   git checkout develop
   git pull origin develop
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/my-feature
   ```

3. **Make Changes**
   - Write code
   - Write tests
   - Update documentation

4. **Run Tests Locally**
   ```bash
   pytest
   ```

5. **Check Code Quality**
   ```bash
   # Format code
   black backend/ frontend/

   # Lint code
   ruff check backend/ frontend/

   # Type check
   mypy backend/

   # Security check
   bandit -r backend/
   ```

6. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

7. **Push and Create PR**
   ```bash
   git push origin feature/my-feature
   # Create PR on GitHub
   ```

### Pre-commit Hooks

Hooks run automatically on `git commit`:
- Trailing whitespace removal
- YAML/JSON validation
- Black formatting
- Ruff linting
- Mypy type checking
- Bandit security check

To run manually:
```bash
pre-commit run --all-files
```

To skip hooks (not recommended):
```bash
git commit --no-verify
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov=frontend --cov-report=html

# Run specific test file
pytest tests/backend/test_api.py

# Run specific test
pytest tests/backend/test_api.py::test_health_check

# Run with verbose output
pytest -v

# Run failed tests only
pytest --lf

# Run in parallel (with pytest-xdist)
pytest -n auto
```

### Writing Tests

#### Backend Test Example

```python
# tests/backend/test_api.py
import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
```

#### Integration Test Example

```python
# tests/integration/test_end_to_end.py
import pytest
import requests

@pytest.fixture
def backend_url():
    return "http://localhost:8000"

def test_new_client_prediction(backend_url):
    payload = {
        "EXT_SOURCE_1": 0.5,
        # ... other fields
    }

    response = requests.post(
        f"{backend_url}/api/v1/predict/new",
        json=payload
    )

    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert 0 <= data["probability"] <= 1
```

### Test Coverage

View coverage report:
```bash
pytest --cov=backend --cov-report=html
open htmlcov/index.html
```

Aim for >70% coverage.

---

## Debugging

### Backend Debugging

#### VS Code Launch Configuration

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Backend",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/backend/main.py",
      "console": "integratedTerminal",
      "env": {
        "MODEL_PATH": "model_saved.pkl",
        "DATA_PATH": "reduced_train.csv",
        "LOG_LEVEL": "DEBUG"
      }
    }
  ]
}
```

#### Python Debugger

```python
# Insert breakpoint
import pdb; pdb.set_trace()

# Or use breakpoint() (Python 3.7+)
breakpoint()
```

### Frontend Debugging

#### Streamlit Debugging

```python
# Display variable
st.write("Debug:", variable)

# Display dataframe
st.dataframe(df)

# Display JSON
st.json(data)
```

### Docker Debugging

```bash
# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Execute command in container
docker-compose exec backend bash
docker-compose exec frontend bash

# Check container status
docker-compose ps

# Inspect container
docker inspect <container_id>
```

---

## Code Quality

### Formatting with Black

```bash
# Format all code
black backend/ frontend/ config/

# Check without modifying
black --check backend/ frontend/

# Format specific file
black backend/main.py
```

### Linting with Ruff

```bash
# Lint all code
ruff check backend/ frontend/ config/

# Auto-fix issues
ruff check --fix backend/

# Show rule documentation
ruff rule E501
```

### Type Checking with Mypy

```bash
# Type check backend
mypy backend/

# Strict mode
mypy --strict backend/

# Generate type stubs
stubgen -p backend -o stubs
```

### Security Scanning with Bandit

```bash
# Scan backend
bandit -r backend/

# Generate report
bandit -r backend/ -f json -o bandit-report.json

# Exclude tests
bandit -r backend/ --exclude tests/
```

---

## Common Tasks

### Adding a New API Endpoint

1. **Define Pydantic Schema**
   ```python
   # backend/schemas/prediction.py
   class NewRequest(BaseModel):
       field: str
   ```

2. **Create Route Handler**
   ```python
   # backend/api/routes.py
   @router.post("/api/v1/new-endpoint")
   async def new_endpoint(request: NewRequest):
       # Implementation
       return {"result": "success"}
   ```

3. **Write Tests**
   ```python
   # tests/backend/test_api.py
   def test_new_endpoint(client):
       response = client.post("/api/v1/new-endpoint", json={"field": "value"})
       assert response.status_code == 200
   ```

4. **Update Documentation**
   - Add to `docs/API.md`
   - Update OpenAPI schema

### Adding a New Frontend Page

1. **Create Page Function**
   ```python
   # frontend/ui.py
   def new_page():
       st.markdown("# New Page")
       # Page implementation
   ```

2. **Add to Navigation**
   ```python
   page_names_to_funcs = {
       "Home": home_page,
       "New Page": new_page,
   }
   ```

### Updating Dependencies

```bash
# Update single package
pip install --upgrade package-name

# Update all packages
pip install --upgrade -r backend/requirements.txt

# Check for outdated packages
pip list --outdated

# Generate new requirements
pip freeze > requirements.txt
```

### Adding Environment Variable

1. **Update `.env.example`**
   ```bash
   NEW_VARIABLE=default_value
   ```

2. **Update `config/settings.py`**
   ```python
   new_variable: str = Field(default="default", env="NEW_VARIABLE")
   ```

3. **Use in Code**
   ```python
   from config.settings import settings
   value = settings.new_variable
   ```

---

## Troubleshooting

### Common Issues

#### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'backend'`

**Solution**:
```bash
# Ensure parent directory is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add to main.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
```

#### Docker Build Fails

**Problem**: `ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully`

**Solution**:
```bash
# Clear Docker cache
docker-compose build --no-cache

# Check requirements.txt for issues
cat backend/requirements.txt

# Try building manually
cd backend
docker build .
```

#### Model/Data Files Not Found

**Problem**: `FileNotFoundError: model_saved.pkl`

**Solution**:
```bash
# Check files exist
ls -lh backend/*.pkl backend/*.csv

# Check environment variables
echo $MODEL_PATH
echo $DATA_PATH

# Ensure correct paths in .env
```

#### Port Already in Use

**Problem**: `Error: Port 8000 is already in use`

**Solution**:
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
export BACKEND_PORT=8001
```

#### Pre-commit Hooks Fail

**Problem**: Hooks fail on commit

**Solution**:
```bash
# Run hooks manually to see errors
pre-commit run --all-files

# Fix formatting issues
black .

# Fix linting issues
ruff check --fix .

# Update hooks
pre-commit autoupdate
```

---

## Performance Tips

### Profiling

```bash
# Profile backend
python -m cProfile -o profile.stats backend/main.py

# Analyze profile
python -m pstats profile.stats

# Profile with py-spy (sampling profiler)
py-spy record -o profile.svg -- python backend/main.py
```

### Memory Profiling

```bash
# Install memory-profiler
pip install memory-profiler

# Profile function
@profile
def my_function():
    # code

# Run profiler
python -m memory_profiler script.py
```

---

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)

---

## Getting Help

- 📧 Open an issue on GitHub
- 💬 Start a discussion
- 📖 Check documentation in `docs/`
- 🔍 Search closed issues/PRs

Happy coding! 🚀
