# Contributing to Credit Scoring Application

Thank you for considering contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git
- Basic understanding of FastAPI and Streamlit

### Setting Up Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/credit-risk-api.git
   cd credit-risk-api
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   pip install -r frontend/requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your local configuration
   ```

6. **Download data files**
   See `DATA_README.md` for instructions on obtaining model and data files.

## Development Workflow

### Branching Strategy

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Urgent production fixes

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Making Changes

1. Make your changes in small, logical commits
2. Write/update tests for your changes
3. Update documentation as needed
4. Run tests locally
5. Ensure all pre-commit hooks pass

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (configured in pyproject.toml)
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Required for all function signatures
- **Imports**: Organized with `isort`

### Code Quality Tools

All code must pass these checks:

```bash
# Format code
black backend/ frontend/ config/

# Lint code
ruff check backend/ frontend/ config/

# Type check
mypy backend/ --ignore-missing-imports

# Security check
bandit -r backend/
```

Pre-commit hooks will run these automatically on commit.

### Docstring Example

```python
def predict_client(client_id: int) -> float:
    """
    Predict default probability for a client.

    Args:
        client_id: Unique identifier for the client (row index).

    Returns:
        Probability of loan default (0.0 to 1.0).

    Raises:
        ValueError: If client_id is invalid or out of range.
        FileNotFoundError: If training data is not available.

    Example:
        >>> predictor = ModelPredictor("model.pkl", "data.csv")
        >>> probability = predictor.predict_client(100)
        >>> print(f"Default risk: {probability:.2%}")
        Default risk: 23.45%
    """
    # Implementation here
    pass
```

## Testing Guidelines

### Running Tests

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

### Writing Tests

- Place tests in `tests/` directory mirroring the source structure
- Test file names must start with `test_`
- Test functions must start with `test_`
- Use fixtures for common setup
- Aim for >70% code coverage

### Test Example

```python
import pytest
from backend.models.predictor import ModelPredictor


@pytest.fixture
def predictor():
    """Create a predictor instance for testing."""
    return ModelPredictor("model.pkl", "data.csv")


def test_predict_new_client(predictor):
    """Test prediction for new client."""
    data = {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        # ... other fields
    }

    result = predictor.predict_new_client(data)

    assert 0.0 <= result <= 1.0
    assert isinstance(result, float)


def test_invalid_client_id(predictor):
    """Test error handling for invalid client ID."""
    with pytest.raises(ValueError, match="Invalid client_id"):
        predictor.predict_existing_client(999999)
```

## Commit Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Examples

```bash
feat(api): add batch prediction endpoint

Implement new endpoint for predicting multiple clients at once.
Improves throughput by 300% compared to individual requests.

Closes #123

---

fix(predictor): handle missing values in input data

Previously, missing values would cause crashes. Now they are
handled gracefully with appropriate error messages.

Fixes #456

---

docs(readme): update deployment instructions

Add section on environment variables and health checks.
```

### Commit Message Rules

- Use present tense ("add feature" not "added feature")
- Use imperative mood ("move cursor to..." not "moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs in footer

## Pull Request Process

### Before Submitting

1. ✅ All tests pass locally
2. ✅ Code follows style guidelines
3. ✅ Documentation is updated
4. ✅ Commit messages follow guidelines
5. ✅ Branch is up to date with develop
6. ✅ No merge conflicts

### Submitting a PR

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request on GitHub**
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Link related issues
   - Add appropriate labels

3. **PR Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   Describe testing performed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Comments added for complex code
   - [ ] Documentation updated
   - [ ] No new warnings generated
   - [ ] Tests added/updated
   - [ ] All tests pass
   - [ ] Dependent changes merged

   ## Screenshots (if applicable)

   ## Related Issues
   Closes #issue_number
   ```

### Review Process

1. At least one maintainer review required
2. All CI/CD checks must pass
3. No unresolved conversations
4. Branch must be up to date with base branch

### After Merge

1. Delete your feature branch
2. Pull latest develop
3. Update local branches

## Project Structure

Understanding the project structure helps with contributions:

```
.
├── backend/                 # FastAPI backend
│   ├── api/                # API routes and middleware
│   │   ├── routes.py       # Endpoint definitions
│   │   └── middleware.py   # Security middleware
│   ├── models/             # ML model wrapper
│   │   └── predictor.py    # ModelPredictor class
│   ├── schemas/            # Pydantic models
│   │   └── prediction.py   # Request/response schemas
│   ├── utils/              # Utilities
│   │   └── logger.py       # Logging configuration
│   └── main.py             # Application entry point
├── frontend/               # Streamlit frontend
│   ├── utils/              # Utilities
│   │   └── api_client.py   # Backend API client
│   └── ui.py               # Main UI application
├── config/                 # Configuration
│   └── settings.py         # Settings management
├── tests/                  # Test suite
│   ├── backend/            # Backend tests
│   └── integration/        # Integration tests
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## Questions or Need Help?

- 📧 Open an issue for bugs or feature requests
- 💬 Start a discussion for questions
- 📖 Check existing documentation

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

---

Thank you for contributing! 🎉
