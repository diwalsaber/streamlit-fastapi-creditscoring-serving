# API Documentation

Complete reference for the Credit Scoring API.

## Base URL

- **Production**: `https://your-domain.com`
- **Development**: `http://localhost:8000`
- **Docker**: `http://backend:8000` (from frontend container)

## Authentication

Currently, the API does not require authentication. For production deployment, consider implementing:
- API keys
- JWT tokens
- OAuth 2.0

## API Versioning

Current version: **v1**

All endpoints are prefixed with `/api/v1/`.

## Endpoints

### Health Check

#### GET `/health`

Check the health status of the API and its dependencies.

**Response**

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "model_loaded": true,
  "data_loaded": true
}
```

**Status Codes**
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is unhealthy

---

### Predict New Client

#### POST `/api/v1/predict/new`

Predict loan default probability for a new client.

**Request Body**

```json
{
  "EXT_SOURCE_1": 0.5,
  "EXT_SOURCE_2": 0.6,
  "EXT_SOURCE_3": 0.4,
  "DAYS_BIRTH": -15000.0,
  "AMT_GOODS_PRICE": 450000.0,
  "AMT_CREDIT": 500000.0,
  "AMT_ANNUITY": 25000.0,
  "DAYS_EMPLOYED": -2000.0,
  "CODE_GENDER": 1.0,
  "AMT_INCOME_TOTAL": 150000.0,
  "DAYS_EMPLOYED_PERC": 0.13,
  "INCOME_CREDIT_PERC": 0.30,
  "ANNUITY_INCOME_PERC": 0.17,
  "PAYMENT_RATE": 0.05
}
```

**Field Descriptions**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `EXT_SOURCE_1` | float | 0.0-1.0 | External source score 1 |
| `EXT_SOURCE_2` | float | 0.0-1.0 | External source score 2 |
| `EXT_SOURCE_3` | float | 0.0-1.0 | External source score 3 |
| `DAYS_BIRTH` | float | negative | Client's age in days (negative) |
| `AMT_GOODS_PRICE` | float | ≥0 | Price of goods for the loan |
| `AMT_CREDIT` | float | ≥0 | Credit amount of the loan |
| `AMT_ANNUITY` | float | ≥0 | Loan annuity |
| `DAYS_EMPLOYED` | float | negative | Days before application person started employment |
| `CODE_GENDER` | float | 0 or 1 | Gender (0=Male, 1=Female) |
| `AMT_INCOME_TOTAL` | float | ≥0 | Income of the client |
| `DAYS_EMPLOYED_PERC` | float | - | Days employed percentage |
| `INCOME_CREDIT_PERC` | float | - | Income to credit percentage |
| `ANNUITY_INCOME_PERC` | float | - | Annuity to income percentage |
| `PAYMENT_RATE` | float | - | Payment rate |

**Response**

```json
{
  "probability": 0.23,
  "client_id": null,
  "risk_level": "low"
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `probability` | float | Probability of default (0.0-1.0) |
| `client_id` | int\|null | Client ID (null for new clients) |
| `risk_level` | string | Risk category: "low", "medium", or "high" |

**Risk Level Thresholds**
- **Low**: probability < 0.3
- **Medium**: 0.3 ≤ probability < 0.6
- **High**: probability ≥ 0.6

**Status Codes**
- `200 OK`: Prediction successful
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Prediction failed
- `503 Service Unavailable`: Model not loaded

**Example Request**

```bash
curl -X POST "http://localhost:8000/api/v1/predict/new" \
  -H "Content-Type: application/json" \
  -d '{
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.4,
    "DAYS_BIRTH": -15000,
    "AMT_GOODS_PRICE": 450000,
    "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 25000,
    "DAYS_EMPLOYED": -2000,
    "CODE_GENDER": 1,
    "AMT_INCOME_TOTAL": 150000,
    "DAYS_EMPLOYED_PERC": 0.13,
    "INCOME_CREDIT_PERC": 0.30,
    "ANNUITY_INCOME_PERC": 0.17,
    "PAYMENT_RATE": 0.05
  }'
```

**Python Example**

```python
import requests

url = "http://localhost:8000/api/v1/predict/new"
data = {
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    # ... other fields
}

response = requests.post(url, json=data)
result = response.json()

print(f"Probability: {result['probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

---

### Predict Existing Client

#### POST `/api/v1/predict/existing`

Predict loan default probability for an existing client by ID.

**Request Body**

```json
{
  "id_client": 100
}
```

**Parameters**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `id_client` | int | 0 to N-1 | Client ID (row index in training data) |

**Response**

```json
{
  "probability": 0.35,
  "client_id": 100,
  "risk_level": "medium"
}
```

**Status Codes**
- `200 OK`: Prediction successful
- `404 Not Found`: Client ID not found
- `422 Unprocessable Entity`: Invalid client ID format
- `500 Internal Server Error`: Prediction failed
- `503 Service Unavailable`: Data not loaded

**Example Request**

```bash
curl -X POST "http://localhost:8000/api/v1/predict/existing" \
  -H "Content-Type: application/json" \
  -d '{"id_client": 100}'
```

---

### Get Client Count

#### GET `/api/v1/clients/count`

Get the total number of clients in the training dataset.

**Response**

```json
{
  "count": 307508,
  "valid_range": "0 to 307507"
}
```

**Status Codes**
- `200 OK`: Count retrieved successfully
- `503 Service Unavailable`: Data not loaded

---

## Legacy Endpoints

These endpoints are deprecated but still functional for backward compatibility.
They will be removed in version 3.0.0.

### POST `/predict_new`

⚠️ **Deprecated**: Use `/api/v1/predict/new` instead

Returns only the probability (float) instead of full response object.

### POST `/predict_previous`

⚠️ **Deprecated**: Use `/api/v1/predict/existing` instead

Returns only the probability (float) instead of full response object.

### GET `/home`

⚠️ **Deprecated**: Use `/health` instead

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| 400 | Bad Request | Check request format |
| 404 | Not Found | Verify resource exists |
| 422 | Validation Error | Check field types and ranges |
| 500 | Internal Error | Contact support |
| 503 | Service Unavailable | Service is starting or unhealthy |

### Validation Error Example

```json
{
  "detail": [
    {
      "loc": ["body", "EXT_SOURCE_1"],
      "msg": "ensure this value is less than or equal to 1.0",
      "type": "value_error.number.not_le",
      "ctx": {"limit_value": 1.0}
    }
  ]
}
```

---

## Rate Limiting

Current limit: **60 requests per minute** per IP address.

When rate limit is exceeded:

**Response**
```json
{
  "detail": "Rate limit exceeded: 60 per 1 minute"
}
```

**Status Code**: `429 Too Many Requests`

**Headers**:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Unix timestamp when limit resets

---

## CORS

Allowed origins (configurable via `CORS_ORIGINS` environment variable):
- `http://localhost:8501` (default)
- `http://localhost:3000`
- `http://127.0.0.1:8501`

For production, update to your actual domain.

---

## Interactive Documentation

FastAPI provides interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

---

## Client Libraries

### Python

```python
from dataclasses import dataclass
import requests


@dataclass
class CreditScoringClient:
    base_url: str = "http://localhost:8000"

    def predict_new(self, client_data: dict) -> dict:
        """Predict for new client."""
        response = requests.post(
            f"{self.base_url}/api/v1/predict/new",
            json=client_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def predict_existing(self, client_id: int) -> dict:
        """Predict for existing client."""
        response = requests.post(
            f"{self.base_url}/api/v1/predict/existing",
            json={"id_client": client_id},
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def health_check(self) -> dict:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()


# Usage
client = CreditScoringClient()
result = client.predict_existing(100)
print(f"Probability: {result['probability']:.2%}")
```

### JavaScript/TypeScript

```typescript
interface ClientData {
  EXT_SOURCE_1: number;
  EXT_SOURCE_2: number;
  // ... other fields
}

interface PredictionResponse {
  probability: number;
  client_id: number | null;
  risk_level: "low" | "medium" | "high";
}

class CreditScoringClient {
  constructor(private baseUrl: string = "http://localhost:8000") {}

  async predictNew(data: ClientData): Promise<PredictionResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/predict/new`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!response.ok) throw new Error(`API error: ${response.statusText}`);
    return response.json();
  }

  async predictExisting(clientId: number): Promise<PredictionResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/predict/existing`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id_client: clientId }),
    });

    if (!response.ok) throw new Error(`API error: ${response.statusText}`);
    return response.json();
  }
}
```

---

## Changelog

See [CHANGELOG.md](../CHANGELOG.md) for API version history and breaking changes.

## Support

For issues or questions:
- 📧 Open an issue on GitHub
- 📖 Check the [README](../README.md)
- 💬 Start a discussion
