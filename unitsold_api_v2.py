#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unitsold Prediction API
"""

import os
import sys
import logging
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------
# Helper
# -------------------------

def get_env(name: str, default: str) -> str:
    v = os.environ.get(name, default)
    return v

# -------------------------
# App & Security
# -------------------------

app = FastAPI(title="Unitsold Prediction API", version="1.0.0")

# CORS (allow all)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API_KEY = '33UhpgeznV0NAJdhXkwQy3kTU94_2B9Q2LhKcPoHuYM6Lu15a' # optional
API_KEY = os.environ.get("API_KEY")
@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    if API_KEY:  # enforce only if set
        key = request.headers.get("X-API-Key")
        if key != API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)

# -------------------------
# Paths to artifacts (local assets folder)
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

PATH_PKL = get_env("PATH_PKL", ASSETS_DIR)
MODEL_FILENAME = get_env("MODEL_FILENAME", "lgbm_unitsold_model.pkl")
FEATURES_FILENAME = get_env("FEATURES_FILENAME", "model_features.pkl")
HIST_CSV = get_env("HIST_CSV", os.path.join(ASSETS_DIR, "data_history_for_predict.csv"))

MODEL_PATH = os.path.join(PATH_PKL, MODEL_FILENAME)
FEATURES_PATH = os.path.join(PATH_PKL, FEATURES_FILENAME)

logger = logging.getLogger("unitsold_api")

# -------------------------
# Feature engineering placeholders 
# -------------------------

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["log_date"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["weekday"] = dt.dt.weekday
    df["dayofyear"] = dt.dt.dayofyear
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df

def create_lagged_features(df: pd.DataFrame, lag_days=[1,3,7]) -> pd.DataFrame:
    group_keys = ["asin_y4a", "country", "platform"]
    df = df.sort_values(by=group_keys + ["log_date"])
    if "unitsold" not in df.columns:
        df["unitsold"] = np.nan
    for d in lag_days:
        df[f"lag_{d}"] = df.groupby(group_keys)["unitsold"].shift(d)
    return df

CATEGORICAL_FEATURES_LGBM: List[str] = ['asin_y4a', 'country', 'platform', 'master_category', 'super_category', 'main_category', 'category', 'product_line', 'year', 'month', 'day', 'weekday', 'is_weekend', 'is_big_sale']

# -------------------------
# Load model & data
# -------------------------

try:
    lgbm = joblib.load(MODEL_PATH)
    FEATURES: List[str] = joblib.load(FEATURES_PATH)
    logger.info("Loaded model %s", MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model/features: {e}")

try:
    historical_df = pd.read_csv(HIST_CSV)
    historical_df["log_date"] = pd.to_datetime(historical_df["log_date"])
except Exception as e:
    raise RuntimeError(f"Failed to load historical data: {e}")

# -------------------------
# Core predict
# -------------------------

def _safe_category_cast(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

def predict_unitsold(date, asin_y4a, asp) -> int:
    print(f"Predicting for {asin_y4a} on {date} with ASP={asp}")
    date = pd.to_datetime(date)

    last_day_data = historical_df[
        (historical_df["asin_y4a"] == asin_y4a) &
        (historical_df["log_date"] < date)
    ].sort_values(by="log_date", ascending=False).drop_duplicates(
        subset=["asin_y4a","country","platform"]
    )

    if last_day_data.empty:
        return 0

    drop_cols = {"direct_price_gap","port_price_gap","year","month","day","weekday","dayofyear","is_weekend","y4a_asp"}
    base_cols = [c for c in FEATURES if not c.startswith("lag_") and c not in drop_cols]

    predict_df = last_day_data[base_cols].copy()
    predict_df["log_date"] = date
    predict_df["y4a_asp"] = asp

    predict_df = create_time_features(predict_df.copy())
    combined_df = pd.concat([historical_df, predict_df], ignore_index=True)
    combined_df = combined_df.sort_values(by=["asin_y4a","country","platform","log_date"]).reset_index(drop=True)

    predict_df_with_lag = create_lagged_features(combined_df.copy(), lag_days=[1,3,7])
    predict_df = predict_df_with_lag[predict_df_with_lag["log_date"] == date].copy()

    for col in ["direct_comp_asp","port_comp_asp"]:
        if col not in predict_df.columns:
            predict_df[col] = np.nan

    predict_df["direct_price_gap"] = predict_df["y4a_asp"] - predict_df["direct_comp_asp"].fillna(-999)
    predict_df["port_price_gap"]  = predict_df["y4a_asp"] - predict_df["port_comp_asp"].fillna(-999)

    X_predict = predict_df[FEATURES].copy()
    X_predict = _safe_category_cast(X_predict, CATEGORICAL_FEATURES_LGBM)

    y_pred_log = lgbm.predict(X_predict)
    y_pred = np.expm1(y_pred_log)
    y_pred[y_pred < 0] = 0
    return int(y_pred[0])

# -------------------------
# Schemas & Routes
# -------------------------


class PredictIn(BaseModel):
    asin_y4a: str
    asp: float


class PredictOut(BaseModel):
    asin_y4a: str
    date: str
    asp: float
    unitsold: int

class BatchPredictIn(BaseModel):
    items: List[PredictIn]

@app.get("/health")
def health():
    return {"status":"ok","model":MODEL_FILENAME,"n_features":len(FEATURES)}

@app.get("/features")
def features():
    return {"features": FEATURES}


@app.post("/predict", response_model=PredictOut)
def predict_endpoint(payload: PredictIn):
    try:
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        y = predict_unitsold(today, payload.asin_y4a, payload.asp)
        return PredictOut(asin_y4a=payload.asin_y4a, date=today, asp=payload.asp, unitsold=y)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(payload: BatchPredictIn):
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    out: List[PredictOut] = []
    for item in payload.items:
        try:
            y = predict_unitsold(today, item.asin_y4a, item.asp)
            out.append(PredictOut(asin_y4a=item.asin_y4a, date=today, asp=item.asp, unitsold=y))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{item.asin_y4a}: {e}")
    return out

# -------------------------
# Main function
# -------------------------

def run_uvicorn(host: str, port: int, log_level: str = "info"):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())

def maybe_start_ngrok(token: str, port: int) -> str:
    if not token:
        return ""
    try:
        from pyngrok import ngrok, conf
        conf.get_default().auth_token = token
        public_url = ngrok.connect(addr=port, proto="http").public_url
        logger.info("Public URL: %s", public_url)
        logger.info("Docs: %s/docs", public_url)
        logger.info("OpenAPI: %s/openapi.json", public_url)
        return public_url
    except Exception as e:
        logger.error("Failed to start ngrok: %s", e)
        return ""

def main(host="0.0.0.0", port=8000, use_ngrok=False, ngrok_token=None, log_level="info"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if use_ngrok and ngrok_token:
        public_url = maybe_start_ngrok(ngrok_token, port)
        print("Public URL:", public_url) 

    run_uvicorn(host, port, log_level)

if __name__ == "__main__":
    # Run local
    # main(host="127.0.0.1", port=8000)

    # Run public with ngrok
    from config import config
    token_ngrok = config.token
    # create account on ngrok.com to get token

    main(host="0.0.0.0", port=8000, use_ngrok=True, ngrok_token=token_ngrok)

