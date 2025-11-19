# Data and Model Files

Due to their size, the following files are **NOT** included in the Git repository:

## Backend Files
- `backend/model_saved.pkl` (757 KB) - Trained LightGBM model
- `backend/reduced_train.csv` (47 MB) - Training data for previous client predictions

## Frontend Files
- `frontend/explainer.pkl` (1.9 MB) - SHAP explainer
- `frontend/shap_values.pkl` (14 MB) - Pre-computed SHAP values
- `frontend/result_for_plot.csv` (47 MB) - Data for plotting

## How to Get These Files

### Option 1: Download from Cloud Storage (Recommended)
If you have access to the cloud storage bucket:
```bash
./scripts/download_data.sh
```

### Option 2: Use Existing Files
If you already have these files locally, place them in the appropriate directories:
- Backend files → `backend/`
- Frontend files → `frontend/`

### Option 3: Generate from Source
If you have the original training data and model training scripts:
```bash
# Train the model and generate required files
python scripts/train_model.py
python scripts/generate_shap_values.py
```

## For Production Deployment

In production, these files should be:
1. Stored in cloud storage (S3, GCS, Azure Blob)
2. Downloaded during container startup
3. Cached appropriately
4. Versioned for model tracking

See `scripts/download_data.sh` for an example implementation.
