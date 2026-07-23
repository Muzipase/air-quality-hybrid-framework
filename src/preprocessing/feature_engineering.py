import numpy as np
import pandas as pd
from typing import Optional


def categorize_aqi(pm25: float, pm10: float) -> str:
    """Create a categorical AQI label from pollution values."""
    try:
        if not np.isfinite(pm25) and np.isfinite(pm10):
            pm25 = pm10 / 2.0
    except Exception:
        pass

    score = pm25 if np.isfinite(pm25) else (pm10 if np.isfinite(pm10) else np.nan)

    if np.isnan(score):
        return "Moderate"
    if score <= 12:
        return "Good"
    if score <= 35.4:
        return "Moderate"
    if score <= 55.4:
        return "Unhealthy"
    if score <= 150.4:
        return "Very Unhealthy"
    return "Hazardous"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add classification labels and derived pollution features."""
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()
    data.columns = [str(col).strip().lower() for col in data.columns]

    if "aqi_category" not in data.columns:
        pm25 = data.get("pm25", pd.Series(dtype=float))
        pm10 = data.get("pm10", pd.Series(dtype=float))
        score = pm25.copy()
        mask_no_pm25 = ~pm25.apply(np.isfinite) if not pm25.empty else pd.Series(dtype=bool)
        score[mask_no_pm25] = pm10[mask_no_pm25] / 2.0

        conditions = [
            ~score.apply(np.isfinite),
            score <= 12,
            score <= 35.4,
            score <= 55.4,
            score <= 150.4,
        ]
        choices = ["Moderate", "Good", "Moderate", "Unhealthy", "Very Unhealthy"]
        data["aqi_category"] = np.select(conditions, choices, default="Hazardous")

    # Add derived features for model expressivity
    if "pm25" in data.columns and "pm10" in data.columns:
        pm10_safe = data["pm10"].replace(0, np.nan)
        data["pm25_pm10_ratio"] = (data["pm25"] / pm10_safe).fillna(0.0)
    else:
        data["pm25_pm10_ratio"] = 0.0

    if "pm25" in data.columns and "o3" in data.columns:
        data["pollution_load"] = data["pm25"] + data["o3"]
    else:
        data["pollution_load"] = data.get("pm25", 0.0) + data.get("o3", 0.0)

    data["is_high_pm25"] = data.get("pm25", 0.0) > 35.4
    data["is_high_pm10"] = data.get("pm10", 0.0) > 150.0

    # Temporal features from timestamp
    if "timestamp" in data.columns:
        ts = pd.to_datetime(data["timestamp"], errors="coerce")
        data["hour"] = ts.dt.hour
        data["day_of_week"] = ts.dt.dayofweek
        data["month"] = ts.dt.month
    else:
        data["hour"] = 0
        data["day_of_week"] = 0
        data["month"] = 1

    return data
