"""
FastAPI Backend for Air Quality Hybrid Framework
Serves as REST API endpoints for separating backend ML logic from frontend UI
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import time
import importlib.util

psutil = None
if importlib.util.find_spec("psutil") is not None:
    psutil = importlib.import_module("psutil")

# Import ML modules from src
from src.ingestion.fetch_data import fetch_openaq_data, fetch_data
from src.ingestion.data_validator import validate_dataframe
from src.preprocessing.clean_data import clean_data
from src.preprocessing.missing_values import fill_missing_values
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.scaling import fit_scaler, transform_with_scaler, save_scaler, load_scaler, apply_scaler_to_dataframe
from src.preprocessing.split_dataset import split_data
from src.models.baseline_svm import train_baseline_svm
from src.models.optimized_svm import train_optimized_svm
from src.evaluation.metrics import compute_metrics
from src.evaluation.confusion_matrix import generate_confusion_matrix
from src.evaluation.cross_validation import cross_validate_model
from src.explainability.shap_explainer import ShapExplainer
from src.balancing.imbalance_analysis import analyze_imbalance
from src.balancing.smote_tomek import apply_smote_tomek
from config.paths import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    BASELINE_MODEL_PATH, OPTIMIZED_MODEL_PATH,
    SCALER_PATH, SHAP_VALUES_PATH, METRICS_PATH, ARTIFACTS_DIR, ensure_dirs
)

# Ensure artifact directories exist before saving files
ensure_dirs()


# ==================== In-Memory Cache ====================

class _Cache:
    """Simple in-memory cache with file-mtime invalidation for models, scalers, and DataFrames."""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._scaler: Any = None
        self._scaler_mtime: float = 0.0
        self._processed_df: Optional[pd.DataFrame] = None
        self._processed_mtime: float = 0.0
        self._raw_df: Optional[pd.DataFrame] = None
        self._raw_mtime: float = 0.0
        self._shap_explainer: Any = None
        self._shap_model_mtime: float = 0.0

    # ---- synchronous loaders (called inside to_thread) ----

    def _load_model_sync(self, path: Path, key: str) -> Any:
        mtime = path.stat().st_mtime
        cached = self._models.get(key)
        if cached is not None and self._models.get(f"{key}_mtime") == mtime:
            return cached
        model = joblib.load(path)
        self._models[key] = model
        self._models[f"{key}_mtime"] = mtime
        return model

    def _load_scaler_sync(self) -> Any:
        if not SCALER_PATH.exists():
            return None
        mtime = SCALER_PATH.stat().st_mtime
        if self._scaler is not None and self._scaler_mtime == mtime:
            return self._scaler
        self._scaler = joblib.load(SCALER_PATH)
        self._scaler_mtime = mtime
        return self._scaler

    def _load_processed_df_sync(self) -> Optional[pd.DataFrame]:
        if not PROCESSED_DATA_PATH.exists():
            return None
        mtime = PROCESSED_DATA_PATH.stat().st_mtime
        if self._processed_df is not None and self._processed_mtime == mtime:
            return self._processed_df
        self._processed_df = pd.read_csv(PROCESSED_DATA_PATH)
        self._processed_mtime = mtime
        return self._processed_df

    def _load_raw_df_sync(self) -> Optional[pd.DataFrame]:
        if not RAW_DATA_PATH.exists():
            return None
        mtime = RAW_DATA_PATH.stat().st_mtime
        if self._raw_df is not None and self._raw_mtime == mtime:
            return self._raw_df
        self._raw_df = pd.read_csv(RAW_DATA_PATH)
        self._raw_mtime = mtime
        return self._raw_df

    def _load_shap_explainer_sync(self) -> Any:
        if not OPTIMIZED_MODEL_PATH.exists():
            return None
        model_mtime = OPTIMIZED_MODEL_PATH.stat().st_mtime
        if self._shap_explainer is not None and self._shap_model_mtime == model_mtime:
            return self._shap_explainer
        data = self._load_processed_df_sync()
        if data is None:
            return None
        X = _get_feature_matrix(data)
        model = self._load_model_sync(OPTIMIZED_MODEL_PATH, "optimized")
        self._shap_explainer = ShapExplainer(model, X)
        self._shap_model_mtime = model_mtime
        return self._shap_explainer

    # ---- async wrappers ----

    async def get_model(self, path: Path, key: str) -> Any:
        return await asyncio.to_thread(self._load_model_sync, path, key)

    async def get_scaler(self) -> Any:
        return await asyncio.to_thread(self._load_scaler_sync)

    async def get_processed_df(self) -> Optional[pd.DataFrame]:
        return await asyncio.to_thread(self._load_processed_df_sync)

    async def get_raw_df(self) -> Optional[pd.DataFrame]:
        return await asyncio.to_thread(self._load_raw_df_sync)

    async def get_shap_explainer(self) -> Any:
        return await asyncio.to_thread(self._load_shap_explainer_sync)

    # ---- invalidation ----

    def invalidate_all(self):
        self._models.clear()
        self._scaler = None
        self._scaler_mtime = 0.0
        self._processed_df = None
        self._processed_mtime = 0.0
        self._raw_df = None
        self._raw_mtime = 0.0
        self._shap_explainer = None
        self._shap_model_mtime = 0.0

    def invalidate_models(self):
        self._models.clear()
        self._shap_explainer = None
        self._shap_model_mtime = 0.0

    def invalidate_data(self):
        self._processed_df = None
        self._processed_mtime = 0.0
        self._raw_df = None
        self._raw_mtime = 0.0

    def invalidate_shap(self):
        self._shap_explainer = None
        self._shap_model_mtime = 0.0


cache = _Cache()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
APP_START_TIME = datetime.now()


def _get_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Return only numeric, model-safe feature columns from a processed dataframe."""
    excluded_columns = {"aqi_category", "timestamp", "location", "city", "country"}
    feature_columns = [
        col for col in data.columns if col not in excluded_columns and pd.api.types.is_numeric_dtype(data[col]) and not pd.api.types.is_bool_dtype(data[col])
    ]
    return data[feature_columns]

# Initialize FastAPI app
app = FastAPI(
    title="Air Quality Hybrid Framework API",
    description="REST API for air quality classification using SMOTE-Tomek and Bayesian-Optimized SVM",
    version="1.0.0"
)

# Enable CORS for frontend integration
import os
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Admin token authentication
ADMIN_API_TOKEN = os.environ.get("ADMIN_API_TOKEN", "changeme-admin-token")

def require_admin_token(x_admin_token: str = Header(None)):
    if x_admin_token is None or x_admin_token != ADMIN_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid admin token")
    return True

# ==================== Pydantic Models ====================

class PredictionInput(BaseModel):
    """Model for prediction input"""
    pm25: float
    pm10: float
    no2: float
    so2: float
    co: float
    o3: float
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None

class PredictionResponse(BaseModel):
    """Model for prediction response"""
    prediction: str
    confidence: float
    input_features: Dict[str, float]
    timestamp: str

class MetricsResponse(BaseModel):
    """Model for metrics response"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    per_class_metrics: Dict[str, Dict[str, float]]

class ImbalanceAnalysisResponse(BaseModel):
    """Model for imbalance analysis response"""
    class_distribution: Dict[str, int]
    entropy: float
    gini_coefficient: float
    jensen_shannon_divergence: float

class HealthResponse(BaseModel):
    """Model for health check response"""
    status: str
    timestamp: str
    version: str

class SystemMetricsResponse(BaseModel):
    """Model for system metrics response"""
    cpu_usage: Optional[float]
    memory_usage: Optional[float]
    api_response_time: float
    uptime: float
    status: str
    timestamp: str

# ==================== Health & Status Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/status")
async def get_status():
    """Get detailed API status"""
    try:
        baseline_exists = BASELINE_MODEL_PATH.exists()
        optimized_exists = OPTIMIZED_MODEL_PATH.exists()
        data_exists = RAW_DATA_PATH.exists()
        
        return {
            "status": "operational",
            "models": {
                "baseline_svm": "trained" if baseline_exists else "not trained",
                "optimized_svm": "trained" if optimized_exists else "not trained"
            },
            "data": {
                "raw_data": "available" if data_exists else "not available"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """Return runtime system metrics for UI health and monitoring."""
    start_time = time.perf_counter()
    cpu_usage = None
    memory_usage = None

    if psutil is not None:
        try:
            cpu_usage = await asyncio.to_thread(psutil.cpu_percent, 0.1)
            memory_usage = await asyncio.to_thread(lambda: psutil.virtual_memory().percent)
        except Exception as exc:
            logger.warning(f"Failed to gather psutil metrics: {exc}")

    uptime_seconds = (datetime.now() - APP_START_TIME).total_seconds()
    response_time_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "api_response_time": round(response_time_ms, 2),
        "uptime": round(uptime_seconds, 2),
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }

# ==================== Data Endpoints ====================

@app.get("/api/data/fetch")
async def fetch_data_endpoint(
    source: str = "auto",
    city: Optional[str] = None,
    forecast_days: int = 1,
):
    """Fetch fresh air quality data from OpenAQ or Open-Meteo, optionally filtered by city.

    Parameters
    ----------
    source : str
        ``openaq``, ``openmeteo``, or ``auto`` (default tries Open-Meteo first, then OpenAQ).
    city : str, optional
        Filter to a specific city (e.g. Lusaka, Kitwe, Ndola).
    forecast_days : int
        Number of forecast days for Open-Meteo (default 1 = today only).
    """
    try:
        logger.info("Fetching data via %s for city: %s...", source, city or "all")
        data = await asyncio.to_thread(fetch_data, source=source, city=city, forecast_days=forecast_days)

        if data is None or len(data) == 0:
            raise HTTPException(status_code=404, detail=f"No data retrieved from {source}")

        # Save raw data
        await asyncio.to_thread(data.to_csv, RAW_DATA_PATH, index=False)
        cache.invalidate_data()
        logger.info("Data fetched successfully: %d records from %s", len(data), city or "all cities")

        return {
            "status": "success",
            "source": source,
            "records_count": len(data),
            "city": city,
            "columns": list(data.columns),
            "shape": data.shape,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("Data fetch failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/summary")
async def get_data_summary():
    """Get summary statistics of loaded data"""
    try:
        if not RAW_DATA_PATH.exists():
            raise HTTPException(status_code=404, detail="Raw data not found. Use /api/data/fetch first")
        
        data = await cache.get_raw_df()
        if data is None:
            raise HTTPException(status_code=404, detail="Raw data not found. Use /api/data/fetch first")
        
        return {
            "total_records": len(data),
            "total_columns": len(data.columns),
            "columns": list(data.columns),
            "missing_values": data.isnull().sum().to_dict(),
            "data_types": data.dtypes.astype(str).to_dict(),
            "shape": list(data.shape),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Data summary failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- Public read-only endpoints --------------------
@app.get("/public/summary")
async def public_data_summary():
    """Lightweight public summary for dashboards and public site"""
    try:
        if not PROCESSED_DATA_PATH.exists() and not RAW_DATA_PATH.exists():
            return JSONResponse(status_code=200, content={
                "status": "no_data",
                "message": "No data available yet"
            })

        # Prefer processed if available
        if PROCESSED_DATA_PATH.exists():
            df = await cache.get_processed_df()
        else:
            df = await cache.get_raw_df()

        if df is None:
            return JSONResponse(status_code=200, content={
                "status": "no_data",
                "message": "No data available yet"
            })

        # Basic aggregated metrics
        recent = df.tail(100)
        summary = {
            "total_records": len(df),
            "cities_sample": recent['city'].dropna().unique().tolist()[:5] if 'city' in df.columns else [],
            "avg_pm25": float(recent['pm25'].mean()) if 'pm25' in df.columns else None,
            "avg_pm10": float(recent['pm10'].mean()) if 'pm10' in df.columns else None,
            "timestamp": datetime.now().isoformat()
        }

        return JSONResponse(status_code=200, content=summary)
    except Exception as e:
        logger.error(f"Public summary failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/public/health")
async def public_health():
    return JSONResponse(status_code=200, content={
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })


@app.get("/public/city/{city_name}")
async def public_city_air_quality(city_name: str):
    """Return current air quality summary for a specific city (Lusaka, Ndola, Kitwe)."""
    try:
        from src.ingestion.openmeteo_client import (
            get_city_coordinates,
            _fetch_aq_hourly,
            _fetch_weather_hourly,
        )

        coords = get_city_coordinates(city_name)
        if not coords:
            return JSONResponse(status_code=200, content={
                "status": "no_data",
                "city": city_name,
                "message": f"City '{city_name}' not found.",
            })

        entry = coords[0]
        lat, lon = entry["latitude"], entry["longitude"]

        aq_data = await asyncio.to_thread(_fetch_aq_hourly, lat, lon, forecast_days=1)
        wx_data = await asyncio.to_thread(_fetch_weather_hourly, lat, lon, forecast_days=1)

        if not aq_data or "hourly" not in aq_data:
            return JSONResponse(status_code=200, content={
                "status": "no_data",
                "city": city_name,
                "message": f"No air quality data available for {city_name} at this time."
            })

        hourly = aq_data["hourly"]
        times = hourly.get("time", [])

        import zoneinfo as _zi
        from datetime import timezone as _tz
        try:
            city_tz = _zi.ZoneInfo("Africa/Lusaka")
        except Exception:
            city_tz = _tz.utc
        now_local = datetime.now(city_tz)
        current_hour_str = now_local.strftime("%Y-%m-%dT%H:00")

        # Find the index of the current hour
        current_idx = None
        for i, ts in enumerate(times):
            if ts >= current_hour_str:
                current_idx = i
                break
        if current_idx is None:
            current_idx = len(times) - 1

        pm25 = float(hourly.get("pm2_5", [0])[current_idx] or 0)
        pm10 = float(hourly.get("pm10", [0])[current_idx] or 0)
        no2 = float(hourly.get("nitrogen_dioxide", [0])[current_idx] or 0)
        so2 = float(hourly.get("sulphur_dioxide", [0])[current_idx] or 0)
        co = float(hourly.get("carbon_monoxide", [0])[current_idx] or 0)
        o3 = float(hourly.get("ozone", [0])[current_idx] or 0)

        wx_hourly = wx_data.get("hourly", {}) if wx_data else {}
        temp_values = wx_hourly.get("temperature_2m", [])
        humid_values = wx_hourly.get("relative_humidity_2m", [])
        wind_values = wx_hourly.get("wind_speed_10m", [])
        temp = float(temp_values[current_idx] or 0) if current_idx < len(temp_values) else 0
        humidity = float(humid_values[current_idx] or 0) if current_idx < len(humid_values) else 0
        wind_speed = float(wind_values[current_idx] or 0) if current_idx < len(wind_values) else 0

        # Classify air quality
        from src.preprocessing.feature_engineering import categorize_aqi
        category = categorize_aqi(pm25, pm10)

        # Build health advice per category
        advice = {
            "Good": {
                "message": "Air quality is good. Enjoy your day outdoors!",
                "color": "green",
                "action": "Safe for outdoor activities.",
            },
            "Moderate": {
                "message": "Air quality is moderate. Sensitive people should be careful.",
                "color": "yellow",
                "action": "Limit prolonged outdoor exertion if you have breathing problems.",
            },
            "Unhealthy": {
                "message": "Air quality is unhealthy. Everyone may start to experience health effects.",
                "color": "orange",
                "action": "Avoid prolonged outdoor exertion. Keep windows closed.",
            },
            "Very Unhealthy": {
                "message": "Air quality is very unhealthy. Health alert — serious effects possible.",
                "color": "red",
                "action": "Stay indoors. Wear a mask if you must go outside.",
            },
            "Hazardous": {
                "message": "Air quality is hazardous. Emergency conditions!",
                "color": "darkred",
                "action": "Stay indoors. Close all windows. Seek medical help if unwell.",
            },
        }

        cat_advice = advice.get(category, advice["Moderate"])

        return JSONResponse(status_code=200, content={
            "status": "ok",
            "city": city_name.title(),
            "category": category,
            "health": cat_advice,
            "readings": {
                "pm25": round(pm25, 1),
                "pm10": round(pm10, 1),
                "no2": round(no2, 1),
                "so2": round(so2, 1),
                "co": round(co, 1),
                "o3": round(o3, 1),
                "temperature": round(temp, 1),
                "humidity": round(humidity, 1),
                "wind_speed": round(wind_speed, 1),
            },
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"City air quality failed for {city_name}: {str(e)}")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "city": city_name.title(),
            "message": f"Could not fetch air quality data for {city_name}.",
        })


def _pm25_to_aqi(pm25: float) -> int:
    """Convert PM2.5 concentration to US AQI value using EPA breakpoints."""
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.0,  35.4,  50,  100),
        (35.4,  55.4, 100,  150),
        (55.4, 150.4, 150,  200),
        (150.4, 250.4, 200, 300),
        (250.4, 500.4, 300, 500),
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if pm25 <= c_high:
            return round(i_low + ((pm25 - c_low) / (c_high - c_low)) * (i_high - i_low))
    return 500


def _aqi_category(aqi: int) -> str:
    """Return category string for an AQI value."""
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy"
    if aqi <= 200:
        return "Very Unhealthy"
    return "Hazardous"


@app.get("/public/city/{city_name}/forecast")
async def public_city_forecast(city_name: str, forecast_days: int = 3):
    """Return hourly + daily air quality forecast for a city."""
    try:
        from src.ingestion.openmeteo_client import (
            get_city_coordinates,
            _fetch_aq_hourly,
            _fetch_weather_hourly,
        )

        coords = get_city_coordinates(city_name)
        if not coords:
            return JSONResponse(status_code=404, content={
                "status": "error",
                "message": f"City '{city_name}' not found.",
            })

        entry = coords[0]
        lat, lon = entry["latitude"], entry["longitude"]

        aq_data = await asyncio.to_thread(_fetch_aq_hourly, lat, lon, forecast_days=forecast_days)
        wx_data = await asyncio.to_thread(_fetch_weather_hourly, lat, lon, forecast_days=forecast_days)

        if not aq_data or "hourly" not in aq_data:
            return JSONResponse(status_code=200, content={
                "status": "no_data",
                "city": city_name.title(),
                "hourly": [],
                "daily": [],
            })

        aq_hourly = aq_data["hourly"]
        times = aq_hourly.get("time", [])
        pm25_values = aq_hourly.get("pm2_5", [])

        wx_hourly = wx_data.get("hourly", {}) if wx_data else {}
        temp_values = wx_hourly.get("temperature_2m", [])
        humid_values = wx_hourly.get("relative_humidity_2m", [])
        wind_values = wx_hourly.get("wind_speed_10m", [])

        # Determine current hour in the city's timezone to filter past hours
        from datetime import timezone as tz
        import zoneinfo as _zi
        try:
            city_tz = _zi.ZoneInfo("Africa/Lusaka")
        except Exception:
            city_tz = tz.utc
        now_local = datetime.now(city_tz)
        current_hour_str = now_local.strftime("%Y-%m-%dT%H:00")

        hourly = []
        daily_map = {}

        for i, ts in enumerate(times):
            # Skip hours that are before the current hour
            if ts < current_hour_str:
                # Still accumulate daily stats for today's past hours
                pm25_d = float(pm25_values[i]) if i < len(pm25_values) and pm25_values[i] is not None else 0
                aqi_d = _pm25_to_aqi(pm25_d)
                temp_d = round(float(temp_values[i])) if i < len(temp_values) and temp_values[i] is not None else 0
                wind_d = round(float(wind_values[i])) if i < len(wind_values) and wind_values[i] is not None else 0
                humid_d = round(float(humid_values[i])) if i < len(humid_values) and humid_values[i] is not None else 0
                date_str = ts[:10]
                if date_str not in daily_map:
                    daily_map[date_str] = {"date": date_str, "aqi_values": [], "temps": [], "winds": [], "humids": []}
                daily_map[date_str]["aqi_values"].append(aqi_d)
                daily_map[date_str]["temps"].append(temp_d)
                daily_map[date_str]["winds"].append(wind_d)
                daily_map[date_str]["humids"].append(humid_d)
                continue
            pm25 = float(pm25_values[i]) if i < len(pm25_values) and pm25_values[i] is not None else 0
            temp = round(float(temp_values[i])) if i < len(temp_values) and temp_values[i] is not None else 0
            humid = round(float(humid_values[i])) if i < len(humid_values) and humid_values[i] is not None else 0
            wind = round(float(wind_values[i])) if i < len(wind_values) and wind_values[i] is not None else 0

            aqi = _pm25_to_aqi(pm25)
            cat = _aqi_category(aqi)

            hourly.append({
                "timestamp": ts,
                "aqi": aqi,
                "pm25": round(pm25, 1),
                "temperature": temp,
                "humidity": humid,
                "wind_speed": wind,
                "category": cat,
            })

            date_str = ts[:10]
            if date_str not in daily_map:
                daily_map[date_str] = {
                    "date": date_str,
                    "aqi_values": [],
                    "temps": [],
                    "winds": [],
                    "humids": [],
                }
            daily_map[date_str]["aqi_values"].append(aqi)
            daily_map[date_str]["temps"].append(temp)
            daily_map[date_str]["winds"].append(wind)
            daily_map[date_str]["humids"].append(humid)

        daily = []
        for date_str, d in daily_map.items():
            aqi_max = max(d["aqi_values"])
            daily.append({
                "date": date_str,
                "aqi_max": aqi_max,
                "aqi_min": min(d["aqi_values"]),
                "temp_max": max(d["temps"]),
                "temp_min": min(d["temps"]),
                "wind_max": max(d["winds"]),
                "humidity_avg": round(sum(d["humids"]) / len(d["humids"])),
                "category": _aqi_category(aqi_max),
            })

        return JSONResponse(status_code=200, content={
            "status": "ok",
            "city": city_name.title(),
            "hourly": hourly,
            "daily": daily,
            "who_guideline": {
                "pm25_annual": 5,
                "pm25_24h": 15,
            },
        })
    except Exception as e:
        logger.error(f"Forecast failed for {city_name}: {str(e)}")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "city": city_name.title(),
            "message": f"Could not fetch forecast data for {city_name}.",
        })


# ---------------------------------------------------------------------------
# Historical Air Quality
# ---------------------------------------------------------------------------

_POLLUTANT_OPENMETEO_KEYS = {
    "pm25": "pm2_5",
    "pm10": "pm10",
    "no2": "nitrogen_dioxide",
    "so2": "sulphur_dioxide",
    "co": "carbon_monoxide",
    "o3": "ozone",
}


def _chunk_date_range(start: str, end: str, max_days: int = 365):
    """Split a date range into chunks of at most *max_days*."""
    from datetime import timedelta
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    chunks = []
    while s <= e:
        chunk_end = min(s + timedelta(days=max_days - 1), e)
        chunks.append((s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        s = chunk_end + timedelta(days=1)
    return chunks


@app.get("/public/city/{city_name}/historical")
async def public_city_historical(
    city_name: str,
    start_date: str = "2023-01-01",
    end_date: str = "",
    pollutants: str = "pm25,pm10,no2,o3,so2,co",
):
    """Return historical daily-averaged air quality data for a city.

    Query params
    -------------
    start_date : ISO date (yyyy-mm-dd)
    end_date   : ISO date (yyyy-mm-dd), defaults to yesterday
    pollutants : comma-separated list of pollutant keys (pm25, pm10, no2, so2, co, o3)
    """
    try:
        from src.ingestion.openmeteo_client import (
            get_city_coordinates,
            _fetch_aq_historical,
        )

        coords = get_city_coordinates(city_name)
        if not coords:
            return JSONResponse(status_code=200, content={
                "status": "no_data",
                "city": city_name,
                "message": f"City '{city_name}' not found.",
            })

        entry = coords[0]
        lat, lon = entry["latitude"], entry["longitude"]

        if not end_date:
            import zoneinfo as _zi
            try:
                tz = _zi.ZoneInfo("Africa/Lusaka")
            except Exception:
                tz = timezone.utc
            yesterday = datetime.now(tz).date() - __import__("datetime").timedelta(days=1)
            end_date = yesterday.strftime("%Y-%m-%d")

        if start_date > end_date:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": "start_date must be before end_date",
            })

        requested = [p.strip().lower() for p in pollutants.split(",") if p.strip() in _POLLUTANT_OPENMETEO_KEYS]
        if not requested:
            requested = ["pm25", "pm10", "no2"]

        openmeteo_vars = [_POLLUTANT_OPENMETEO_KEYS[p] for p in requested]
        chunks = _chunk_date_range(start_date, end_date)

        all_times, all_data = [], {v: [] for v in openmeteo_vars}

        for cs, ce in chunks:
            raw = await asyncio.to_thread(
                _fetch_aq_historical, lat, lon, cs, ce, openmeteo_vars,
            )
            if raw and "hourly" in raw:
                h = raw["hourly"]
                all_times.extend(h.get("time", []))
                for var in openmeteo_vars:
                    all_data[var].extend(h.get(var, []))

        if not all_times:
            return JSONResponse(status_code=200, content={
                "status": "no_data",
                "city": city_name.title(),
                "message": "No historical data available for the requested period.",
            })

        # --- Aggregate hourly → daily ---
        import numpy as _np

        df = pd.DataFrame({"time": pd.to_datetime(all_times)})
        for var in openmeteo_vars:
            df[var] = pd.to_numeric(all_data[var], errors="coerce")

        df["date"] = df["time"].dt.date
        daily_groups = df.groupby("date")

        daily = []
        for date_val, group in daily_groups:
            row: Dict = {"date": str(date_val)}
            for var in openmeteo_vars:
                target_key = {v: k for k, v in _POLLUTANT_OPENMETEO_KEYS.items()}[var]
                vals = group[var].dropna()
                if len(vals) > 0:
                    row[target_key] = round(float(vals.mean()), 2)
                    row[f"{target_key}_max"] = round(float(vals.max()), 2)
                else:
                    row[target_key] = None
                    row[f"{target_key}_max"] = None
            daily.append(row)

        # --- Summary stats ---
        stats: Dict = {}
        for target_key in requested:
            vals = [d[target_key] for d in daily if d.get(target_key) is not None]
            if vals:
                mean_val = float(_np.mean(vals))
                trend = "stable"
                if len(vals) >= 30:
                    first_half = float(_np.mean(vals[:len(vals) // 2]))
                    second_half = float(_np.mean(vals[len(vals) // 2:]))
                    pct_change = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
                    if pct_change > 10:
                        trend = "increasing"
                    elif pct_change < -10:
                        trend = "decreasing"
                stats[target_key] = {
                    "min": round(float(min(vals)), 2),
                    "max": round(float(max(vals)), 2),
                    "mean": round(mean_val, 2),
                    "trend": trend,
                    "count": len(vals),
                }

        return JSONResponse(status_code=200, content={
            "status": "ok",
            "city": city_name.title(),
            "date_range": {"start": start_date, "end": end_date},
            "pollutants": requested,
            "daily": daily,
            "stats": stats,
        })

    except Exception as e:
        logger.error(f"Historical data failed for {city_name}: {str(e)}")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "city": city_name.title(),
            "message": f"Could not fetch historical data for {city_name}.",
        })


@app.post("/api/data/upload")
async def upload_data(file: UploadFile = File(...), authorized: bool = Depends(require_admin_token)):
    """Upload custom CSV data"""
    try:
        contents = await file.read()
        df = pd.read_csv(b"\n".join(contents.split(b"\n")))
        
        # Validate data
        validation = validate_dataframe(df)
        if not validation.get("is_valid", False):
            raise HTTPException(status_code=400, detail={
                "message": "Uploaded data failed validation",
                "validation": validation
            })
        
        # Save uploaded data
        await asyncio.to_thread(df.to_csv, RAW_DATA_PATH, index=False)
        cache.invalidate_data()
        
        return {
            "status": "success",
            "message": f"File '{file.filename}' uploaded successfully",
            "records": len(df),
            "columns": list(df.columns),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Data upload failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/data/download")
async def download_data(data_type: str = "raw"):
    """Download processed or raw data"""
    try:
        if data_type == "raw":
            file_path = RAW_DATA_PATH
        elif data_type == "processed":
            file_path = PROCESSED_DATA_PATH
        else:
            raise HTTPException(status_code=400, detail="Invalid data_type. Use 'raw' or 'processed'")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"{data_type} data not found")
        
        return FileResponse(
            path=file_path,
            filename=f"air_quality_{data_type}.csv",
            media_type="text/csv"
        )
    except Exception as e:
        logger.error(f"Data download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Preprocessing Endpoints ====================

@app.post("/api/preprocessing/execute")
async def execute_preprocessing(authorized: bool = Depends(require_admin_token)):
    """Execute full preprocessing pipeline"""
    try:
        if not RAW_DATA_PATH.exists():
            raise HTTPException(status_code=404, detail="Raw data not found. Use /api/data/fetch first")
        
        logger.info("Starting preprocessing pipeline...")
        
        # Load raw data
        data = await cache.get_raw_df()
        if data is None:
            raise HTTPException(status_code=404, detail="Raw data not found. Use /api/data/fetch first")
        data = data.copy()
        
        # Clean data
        data = clean_data(data)
        
        # Handle missing values
        data = fill_missing_values(data)
        
        # Engineer features
        data = engineer_features(data)

        # Fit scaler on processed feature data and persist it for future predictions
        NON_FEATURE = {"aqi_category", "timestamp", "location", "city", "country"}
        feature_columns = [col for col in data.columns if col not in NON_FEATURE and pd.api.types.is_numeric_dtype(data[col]) and not pd.api.types.is_bool_dtype(data[col])]
        _, scaler = fit_scaler(data[feature_columns])
        await asyncio.to_thread(save_scaler, scaler, SCALER_PATH)
        data = apply_scaler_to_dataframe(data, scaler, feature_columns)

        # Save processed data
        await asyncio.to_thread(data.to_csv, PROCESSED_DATA_PATH, index=False)
        cache.invalidate_data()
        cache.invalidate_shap()
        
        logger.info(f"Preprocessing completed: {len(data)} records processed")
        
        return {
            "status": "success",
            "message": "Preprocessing completed successfully",
            "records_processed": len(data),
            "output_file": str(PROCESSED_DATA_PATH),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Class Imbalance Analysis Endpoints ====================

@app.get("/api/imbalance/analyze", response_model=ImbalanceAnalysisResponse)
async def analyze_class_imbalance():
    """Analyze class imbalance in the dataset"""
    try:
        if not PROCESSED_DATA_PATH.exists():
            raise HTTPException(status_code=404, detail="Processed data not found. Run preprocessing first")
        
        data = await cache.get_processed_df()
        if data is None:
            raise HTTPException(status_code=404, detail="Processed data not found. Run preprocessing first")
        
        logger.info("Analyzing class imbalance...")
        imbalance_metrics = analyze_imbalance(data)
        
        class_dist = data['aqi_category'].value_counts().to_dict()
        
        return {
            "class_distribution": class_dist,
            "entropy": imbalance_metrics.get("entropy", 0),
            "gini_coefficient": imbalance_metrics.get("gini_coefficient", 0),
            "jensen_shannon_divergence": imbalance_metrics.get("jensen_shannon_divergence", 0)
        }
    except Exception as e:
        logger.error(f"Imbalance analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/imbalance/balance")
async def balance_dataset(authorized: bool = Depends(require_admin_token)):
    """Apply SMOTE-Tomek balancing to training data"""
    try:
        if not PROCESSED_DATA_PATH.exists():
            raise HTTPException(status_code=404, detail="Processed data not found. Run preprocessing first")
        
        data = await cache.get_processed_df()
        if data is None:
            raise HTTPException(status_code=404, detail="Processed data not found. Run preprocessing first")
        data = data.copy()
        
        logger.info("Applying SMOTE-Tomek balancing...")
        
        # Separate features and labels
        X = _get_feature_matrix(data)
        y = data['aqi_category']
        
        # Apply SMOTE-Tomek
        X_balanced, y_balanced = apply_smote_tomek(X, y)
        
        # Combine back
        balanced_data = X_balanced.copy()
        balanced_data['aqi_category'] = y_balanced
        
        logger.info(f"Balancing completed: {len(balanced_data)} records (from {len(data)})")
        
        return {
            "status": "success",
            "message": "SMOTE-Tomek balancing applied successfully",
            "original_records": len(data),
            "balanced_records": len(balanced_data),
            "class_distribution_before": data['aqi_category'].value_counts().to_dict(),
            "class_distribution_after": balanced_data['aqi_category'].value_counts().to_dict(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Balancing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Model Training Endpoints ====================

@app.post("/api/models/train/baseline")
async def train_baseline(background_tasks: BackgroundTasks, authorized: bool = Depends(require_admin_token)):
    """Train baseline SVM model (without optimization)"""
    try:
        if not PROCESSED_DATA_PATH.exists():
            raise HTTPException(status_code=404, detail="Processed data not found. Run preprocessing first")
        
        logger.info("Training baseline SVM model...")
        
        data = await cache.get_processed_df()
        if data is None:
            raise HTTPException(status_code=404, detail="Processed data not found. Run preprocessing first")
        data = data.copy()
        X = _get_feature_matrix(data)
        y = data['aqi_category']
        
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Apply SMOTE-Tomek to training data
        X_train_bal, y_train_bal = apply_smote_tomek(X_train, y_train)
        
        # Train baseline model on balanced data
        model = await asyncio.to_thread(train_baseline_svm, X_train_bal, y_train_bal)
        
        # Save model
        await asyncio.to_thread(joblib.dump, model, BASELINE_MODEL_PATH)
        cache.invalidate_models()
        
        # Calculate metrics on original test set
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, save_path=METRICS_PATH)
        
        logger.info(f"Baseline model trained. Accuracy: {metrics['accuracy']:.4f}")
        
        return {
            "status": "success",
            "message": "Baseline SVM model trained successfully",
            "model_path": str(BASELINE_MODEL_PATH),
            "train_size": len(X_train_bal),
            "test_size": len(X_test),
            "metrics": metrics,
            "metrics_path": str(METRICS_PATH),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Baseline training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/train/optimized")
async def train_optimized(background_tasks: BackgroundTasks, authorized: bool = Depends(require_admin_token)):
    """Train optimized SVM model with Bayesian hyperparameter tuning"""
    try:
        if not PROCESSED_DATA_PATH.exists():
            raise HTTPException(status_code=404, detail="Processed data not found. Run preprocessing first")
        
        logger.info("Training optimized SVM model with Bayesian optimization...")
        
        data = await cache.get_processed_df()
        if data is None:
            raise HTTPException(status_code=404, detail="Processed data not found. Run preprocessing first")
        data = data.copy()
        X = _get_feature_matrix(data)
        y = data['aqi_category']
        
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Apply SMOTE-Tomek to training data
        X_train_bal, y_train_bal = apply_smote_tomek(X_train, y_train)
        
        # Train optimized model on balanced data
        model, best_params, study = await asyncio.to_thread(train_optimized_svm, X_train_bal, y_train_bal)
        
        # Save model
        await asyncio.to_thread(joblib.dump, model, OPTIMIZED_MODEL_PATH)
        cache.invalidate_models()

        # Persist best hyperparameters for reproducibility
        try:
            best_params_path = ARTIFACTS_DIR / "optimized_best_params.json"
            with open(best_params_path, "w", encoding="utf-8") as bp_file:
                json.dump(best_params, bp_file, indent=2)
        except Exception:
            logger.exception("Failed to persist best hyperparameters")
        
        # Calculate metrics on original test set
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, save_path=METRICS_PATH)
        
        logger.info(f"Optimized model trained. Accuracy: {metrics['accuracy']:.4f}")
        
        return {
            "status": "success",
            "message": "Optimized SVM model trained successfully",
            "model_path": str(OPTIMIZED_MODEL_PATH),
            "train_size": len(X_train_bal),
            "test_size": len(X_test),
            "best_hyperparameters": best_params,
            "metrics": metrics,
            "metrics_path": str(METRICS_PATH),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Optimized training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Prediction Endpoints ====================

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput, model_type: str = "optimized"):
    """Make prediction using trained model"""
    try:
        if model_type == "optimized":
            model_path = OPTIMIZED_MODEL_PATH
        elif model_type == "baseline":
            model_path = BASELINE_MODEL_PATH
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type. Use 'optimized' or 'baseline'")
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"{model_type} model not trained. Train model first")
        
        # Load model from cache
        model = await cache.get_model(model_path, model_type)
        
        # Prepare raw input features
        features = {
            'pm25': input_data.pm25,
            'pm10': input_data.pm10,
            'no2': input_data.no2,
            'so2': input_data.so2,
            'co': input_data.co,
            'o3': input_data.o3,
            'temperature': input_data.temperature or 25.0,
            'humidity': input_data.humidity or 60.0,
            'wind_speed': input_data.wind_speed or 5.0
        }

        # Create DataFrame for prediction and apply the same preprocessing logic as training
        input_df = pd.DataFrame([features])
        input_df = clean_data(input_df)
        input_df = fill_missing_values(input_df)
        input_df = engineer_features(input_df)

        feature_columns = [
            col
            for col in input_df.columns
            if col not in ['aqi_category', 'timestamp', 'location', 'city', 'country']
            and pd.api.types.is_numeric_dtype(input_df[col])
            and not pd.api.types.is_bool_dtype(input_df[col])
        ]

        scaler = await cache.get_scaler()
        if scaler is not None:
            scaler_features = list(getattr(scaler, "feature_names_in_", []))
            if scaler_features:
                cols_to_scale = [c for c in scaler_features if c in input_df.columns]
                if cols_to_scale:
                    input_df[cols_to_scale] = transform_with_scaler(input_df[cols_to_scale], scaler)

        input_df = input_df[feature_columns]

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Get confidence (distance from decision boundary)
        decision_function = model.decision_function(input_df)
        abs_max = float(np.max(np.abs(decision_function)))
        confidence = float(np.max(decision_function) / abs_max) if abs_max > 0 else 0.5
        confidence = abs(confidence)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "input_features": features,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/live")
async def predict_live(city: str, model_type: str = "optimized"):
    """Fetch live Open-Meteo readings for a city and classify them with the SVM model."""
    try:
        from src.ingestion.openmeteo_client import (
            get_city_coordinates,
            _fetch_aq_hourly,
            _fetch_weather_hourly,
        )
        from datetime import timezone as tz
        import zoneinfo as _zi

        if model_type == "optimized":
            model_path = OPTIMIZED_MODEL_PATH
        elif model_type == "baseline":
            model_path = BASELINE_MODEL_PATH
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type")

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"{model_type} model not trained. Train model first")

        coords = get_city_coordinates(city)
        if not coords:
            raise HTTPException(status_code=404, detail=f"City '{city}' not found")

        entry = coords[0]
        lat, lon = entry["latitude"], entry["longitude"]

        aq_data = await asyncio.to_thread(_fetch_aq_hourly, lat, lon, forecast_days=1)
        wx_data = await asyncio.to_thread(_fetch_weather_hourly, lat, lon, forecast_days=1)

        if not aq_data or "hourly" not in aq_data:
            raise HTTPException(status_code=503, detail=f"No live air quality data available for {city}")

        hourly = aq_data["hourly"]
        times = hourly.get("time", [])

        try:
            city_tz = _zi.ZoneInfo("Africa/Lusaka")
        except Exception:
            city_tz = tz.utc
        now_local = datetime.now(city_tz)
        current_hour_str = now_local.strftime("%Y-%m-%dT%H:00")

        current_idx = None
        for i, ts in enumerate(times):
            if ts >= current_hour_str:
                current_idx = i
                break
        if current_idx is None:
            current_idx = len(times) - 1

        readings = {
            "pm25": float(hourly.get("pm2_5", [None])[current_idx] or 0),
            "pm10": float(hourly.get("pm10", [None])[current_idx] or 0),
            "no2": float(hourly.get("nitrogen_dioxide", [None])[current_idx] or 0),
            "so2": float(hourly.get("sulphur_dioxide", [None])[current_idx] or 0),
            "co": float(hourly.get("carbon_monoxide", [None])[current_idx] or 0),
            "o3": float(hourly.get("ozone", [None])[current_idx] or 0),
        }

        wx_hourly = wx_data.get("hourly", {}) if wx_data else {}
        temp_values = wx_hourly.get("temperature_2m", [])
        humid_values = wx_hourly.get("relative_humidity_2m", [])
        wind_values = wx_hourly.get("wind_speed_10m", [])
        readings["temperature"] = float(temp_values[current_idx] or 25) if current_idx < len(temp_values) else 25.0
        readings["humidity"] = float(humid_values[current_idx] or 60) if current_idx < len(humid_values) else 60.0
        readings["wind_speed"] = float(wind_values[current_idx] or 5) if current_idx < len(wind_values) else 5.0

        model = await cache.get_model(model_path, model_type)

        input_df = pd.DataFrame([readings])
        input_df = clean_data(input_df)
        input_df = fill_missing_values(input_df)
        input_df = engineer_features(input_df)

        feature_columns = [
            col for col in input_df.columns
            if col not in ['aqi_category', 'timestamp', 'location', 'city', 'country']
            and pd.api.types.is_numeric_dtype(input_df[col])
            and not pd.api.types.is_bool_dtype(input_df[col])
        ]

        scaler = await cache.get_scaler()
        if scaler is not None:
            scaler_features = list(getattr(scaler, "feature_names_in_", []))
            if scaler_features:
                cols_to_scale = [c for c in scaler_features if c in input_df.columns]
                if cols_to_scale:
                    input_df[cols_to_scale] = transform_with_scaler(input_df[cols_to_scale], scaler)

        input_df = input_df[feature_columns]
        prediction = model.predict(input_df)[0]

        decision_function = model.decision_function(input_df)
        abs_max = float(np.max(np.abs(decision_function)))
        confidence = float(np.max(decision_function) / abs_max) if abs_max > 0 else 0.5
        confidence = abs(confidence)

        return {
            "status": "success",
            "city": city.title(),
            "source": "live_open_meteo",
            "prediction": prediction,
            "confidence": confidence,
            "live_readings": {k: round(v, 2) for k, v in readings.items()},
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Live prediction failed for {city}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...), model_type: str = "optimized"):
    """Make batch predictions from uploaded CSV"""
    try:
        model_path = OPTIMIZED_MODEL_PATH if model_type == "optimized" else BASELINE_MODEL_PATH
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"{model_type} model not trained")
        
        # Read uploaded file and apply the same preprocessing pipeline used for training
        contents = await file.read()
        df = pd.read_csv(b"\n".join(contents.split(b"\n")))
        df = clean_data(df)
        df = fill_missing_values(df)
        df = engineer_features(df)

        feature_columns = [
            col
            for col in df.columns
            if col not in ['aqi_category', 'timestamp', 'location', 'city', 'country']
        ]

        scaler = await cache.get_scaler()
        if scaler is not None:
            scaler_features = list(getattr(scaler, "feature_names_in_", []))
            if scaler_features:
                cols_to_scale = [c for c in scaler_features if c in df.columns]
                if cols_to_scale:
                    df[cols_to_scale] = transform_with_scaler(df[cols_to_scale], scaler)

        # Load model from cache
        model = await cache.get_model(model_path, model_type)
        
        # Make predictions
        predictions = model.predict(df[feature_columns])
        
        # Create result dataframe
        result_df = df.copy()
        result_df['prediction'] = predictions
        
        return {
            "status": "success",
            "predictions_count": len(predictions),
            "predictions": predictions.tolist(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Evaluation Endpoints ====================

@app.get("/api/evaluation/metrics")
async def get_model_metrics(model_type: str = "optimized"):
    """Get evaluation metrics for trained model"""
    try:
        if not METRICS_PATH.exists():
            raise HTTPException(status_code=404, detail="Metrics not available. Train and evaluate model first")
        
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        
        return {
            "model_type": model_type,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluation/cross-validate")
async def cross_validate(folds: int = 5, model_type: str = "optimized"):
    """Perform cross-validation on trained model"""
    try:
        if not PROCESSED_DATA_PATH.exists():
            raise HTTPException(status_code=404, detail="Processed data not found")
        
        model_path = OPTIMIZED_MODEL_PATH if model_type == "optimized" else BASELINE_MODEL_PATH
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"{model_type} model not trained")
        
        data = await cache.get_processed_df()
        if data is None:
            raise HTTPException(status_code=404, detail="Processed data not found")
        X = _get_feature_matrix(data)
        y = data['aqi_category']
        
        model = await cache.get_model(model_path, model_type)
        
        # Cross-validate
        cv_results = cross_validate_model(model, X, y, cv=folds)
        
        return {
            "status": "success",
            "model_type": model_type,
            "folds": folds,
            "cv_results": cv_results,
            "mean_accuracy": float(np.mean(cv_results['test_accuracy'])) if cv_results['test_accuracy'] else 0.0,
            "std_accuracy": float(np.std(cv_results['test_accuracy'])) if cv_results['test_accuracy'] else 0.0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cross-validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Explainability Endpoints ====================

@app.get("/api/explainability/shap-summary")
async def get_shap_summary():
    """Get SHAP feature importance summary"""
    try:
        if not OPTIMIZED_MODEL_PATH.exists():
            raise HTTPException(status_code=404, detail="Optimized model not trained. Train model first")
        
        logger.info("Computing SHAP values...")
        
        explainer = await cache.get_shap_explainer()
        if explainer is None:
            raise HTTPException(status_code=404, detail="Could not build SHAP explainer. Run preprocessing first.")
        
        shap_summary = await asyncio.to_thread(explainer.get_summary)
        
        return {
            "status": "success",
            "shap_summary": shap_summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"SHAP summary failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/explainability/explain-prediction")
async def explain_prediction(input_data: PredictionInput):
    """Get SHAP explanation for a specific prediction"""
    try:
        if not OPTIMIZED_MODEL_PATH.exists():
            raise HTTPException(status_code=404, detail="Model not trained")
        
        explainer = await cache.get_shap_explainer()
        model = await cache.get_model(OPTIMIZED_MODEL_PATH, "optimized")
        if explainer is None or model is None:
            raise HTTPException(status_code=404, detail="Could not build SHAP explainer. Run preprocessing first.")
        
        # Prepare input
        features = {
            'pm25': input_data.pm25,
            'pm10': input_data.pm10,
            'no2': input_data.no2,
            'so2': input_data.so2,
            'co': input_data.co,
            'o3': input_data.o3,
            'temperature': input_data.temperature or 25.0,
            'humidity': input_data.humidity or 60.0,
            'wind_speed': input_data.wind_speed or 5.0
        }
        
        input_df = pd.DataFrame([features])
        input_df = clean_data(input_df)
        input_df = fill_missing_values(input_df)
        input_df = engineer_features(input_df)

        feature_columns = [
            col
            for col in input_df.columns
            if col not in ['aqi_category', 'timestamp', 'location', 'city', 'country']
        ]

        scaler = await cache.get_scaler()
        if scaler is not None:
            scaler_features = list(getattr(scaler, "feature_names_in_", []))
            if scaler_features:
                cols_to_scale = [c for c in scaler_features if c in input_df.columns]
                if cols_to_scale:
                    input_df[cols_to_scale] = transform_with_scaler(input_df[cols_to_scale], scaler)

        input_df = input_df[feature_columns]

        # Get explanation
        explanation = explainer.explain_instance(input_df)
        
        return {
            "status": "success",
            "prediction": model.predict(input_df)[0],
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction explanation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Pipeline Execution Endpoint ====================

@app.post("/api/pipeline/execute")
async def execute_full_pipeline(
    source: str = "auto",
    authorized: bool = Depends(require_admin_token),
):
    """Execute complete ML pipeline from data to model.

    Parameters
    ----------
    source : str
        Data source for step 1: ``openaq``, ``openmeteo``, or ``auto``.
    """
    try:
        logger.info("Starting full ML pipeline execution (source=%s)...", source)

        # Step 1: Fetch data
        logger.info("Step 1: Fetching data via %s...", source)
        data = await asyncio.to_thread(fetch_data, source=source)
        await asyncio.to_thread(data.to_csv, RAW_DATA_PATH, index=False)
        
        # Step 2: Preprocess
        logger.info("Step 2: Preprocessing data...")
        data = clean_data(data)
        data = fill_missing_values(data)
        data = engineer_features(data)
        feature_columns = [col for col in data.columns if col not in ['aqi_category', 'timestamp']]
        scaled_features, scaler = fit_scaler(data[feature_columns])
        await asyncio.to_thread(save_scaler, scaler, SCALER_PATH)
        data[feature_columns] = scaled_features
        await asyncio.to_thread(data.to_csv, PROCESSED_DATA_PATH, index=False)
        
        # Step 3: Analyze imbalance
        logger.info("Step 3: Analyzing class imbalance...")
        imbalance_metrics = analyze_imbalance(data)
        
        # Step 4: Train models
        logger.info("Step 4: Training models...")
        X = _get_feature_matrix(data)
        y = data['aqi_category']
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Apply SMOTE-Tomek to training data before fitting models
        X_train_bal, y_train_bal = apply_smote_tomek(X_train, y_train)
        
        baseline_model = await asyncio.to_thread(train_baseline_svm, X_train_bal, y_train_bal)
        optimized_model, best_params, _ = await asyncio.to_thread(train_optimized_svm, X_train_bal, y_train_bal)
        
        await asyncio.to_thread(joblib.dump, baseline_model, BASELINE_MODEL_PATH)
        await asyncio.to_thread(joblib.dump, optimized_model, OPTIMIZED_MODEL_PATH)
        cache.invalidate_all()
        
        # Persist best_params for reproducibility
        try:
            best_params_path = ARTIFACTS_DIR / "optimized_best_params.json"
            with open(best_params_path, "w", encoding="utf-8") as bp_file:
                json.dump(best_params, bp_file, indent=2)
        except Exception:
            logger.exception("Failed to persist best hyperparameters in pipeline execute")
        
        # Step 5: Evaluate
        logger.info("Step 5: Evaluating models...")
        baseline_metrics = compute_metrics(y_test, baseline_model.predict(X_test))
        optimized_metrics = compute_metrics(y_test, optimized_model.predict(X_test))
        
        logger.info("Pipeline execution completed successfully!")
        
        return {
            "status": "success",
            "message": "Full pipeline executed successfully",
            "steps_completed": 5,
            "data_records": len(data),
            "train_size": len(X_train_bal),
            "test_size": len(X_test),
            "imbalance_metrics": imbalance_metrics,
            "baseline_accuracy": baseline_metrics['accuracy'],
            "optimized_accuracy": optimized_metrics['accuracy'],
            "best_hyperparameters": best_params,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Root Endpoint ====================

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "message": "Air Quality Hybrid Framework API",
        "version": "1.0.0",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
        "endpoints": {
            "health": "/health",
            "data": ["/api/data/fetch?source=openmeteo", "/api/data/summary", "/api/data/upload", "/api/data/download"],
            "preprocessing": ["/api/preprocessing/execute"],
            "imbalance": ["/api/imbalance/analyze", "/api/imbalance/balance"],
            "models": ["/api/models/train/baseline", "/api/models/train/optimized"],
            "prediction": ["/api/predict", "/api/predict/batch"],
            "evaluation": ["/api/evaluation/metrics", "/api/evaluation/cross-validate"],
            "explainability": ["/api/explainability/shap-summary", "/api/explainability/explain-prediction"],
            "system": ["/api/system/metrics"],
            "pipeline": ["/api/pipeline/execute"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
