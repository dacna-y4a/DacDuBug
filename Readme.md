# Unitsold Prediction API

This repository provides a FastAPI-based REST API for predicting units sold using a pre-trained LightGBM model. The API is designed for easy integration and batch/real-time prediction.

## Features
- Predict units sold for a given product (ASIN) and ASP (average selling price)
- Batch prediction endpoint
- Auto-detects current date for prediction (no need to provide date)
- CORS enabled for easy integration
- API key security (optional, via environment variable)
- Easily extensible for new features or models

## Project Structure

```
Project/
├── assets/                  # Model, features, and historical data files (not included in repo)
├── train_model              # File run for train model on google colab
├── unitsold_api_v2.py       # API script (auto-date, improved categorical handling)
├── Readme.md                # This file
├── requirements.txt         # required libraries
```

## Quick Start

1. **Prepare assets**: Place your model (`lgbm_unitsold_model.pkl`), features (`model_features.pkl`), and historical data (`data_history_for_predict.csv`) in the `assets/` folder.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # or manually:
   pip install fastapi uvicorn pandas numpy joblib pyngrok
   ```
3. **Run the API**:
   ```bash
   python unitsold_api_v2.py
   # or
   python unitsold_api.py
   ```
4. **Access the API**:
   - Swagger UI: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

## API Usage

### Predict (Single)
- **Endpoint:** `POST /predict`
- **Request Body:**
  ```json
  {
    "asin_y4a": "B08N5WRWNW",
    "asp": 100.0
  }
  ```
- **Response:**
  ```json
  {
    "asin_y4a": "B08N5WRWNW",
    "date": "2025-10-02",
    "asp": 100.0,
    "unitsold": 42
  }
  ```

### Predict (Batch)
- **Endpoint:** `POST /predict/batch`
- **Request Body:**
  ```json
  {
    "items": [
      {"asin_y4a": "B08N5WRWNW", "asp": 100.0},
      {"asin_y4a": "B07XJ8C8F5", "asp": 120.0}
    ]
  }
  ```
- **Response:** List of prediction results (same format as single predict)

### Health Check
- **Endpoint:** `GET /health`

### Features List
- **Endpoint:** `GET /features`

## Environment Variables
- `API_KEY` (optional): Set to enable API key protection (header: `X-API-Key`)
- `PATH_PKL`, `MODEL_FILENAME`, `FEATURES_FILENAME`, `HIST_CSV`: Override default asset paths if needed

## Notes
- All date handling is automatic (uses server's current date)
- Make sure your assets folder contains the correct files
- For ngrok/public URL, see logs or console output when running with ngrok enabled

## License
MIT

---

> **Note:** Folder `local_handle` is not documented here as requested.
