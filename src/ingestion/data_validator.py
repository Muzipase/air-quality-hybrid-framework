import pandas as pd
from typing import Dict, List

REQUIRED_FEATURE_COLUMNS = ["pm25", "pm10", "no2", "so2", "co", "o3"]
OPTIONAL_FEATURE_COLUMNS = ["temperature", "humidity", "wind_speed", "rainfall"]


def validate_dataframe(df: pd.DataFrame, require_target: bool = False) -> Dict[str, object]:
    """Validate that the dataframe contains required pollutant features and optional weather fields."""
    result = {
        "is_valid": True,
        "missing_columns": [],
        "invalid_columns": [],
        "errors": [],
    }

    if df is None or df.empty:
        result["is_valid"] = False
        result["errors"].append("Dataframe is empty or None")
        return result

    lower_columns = [str(col).strip().lower() for col in df.columns]
    check_df = df.copy()
    check_df.columns = lower_columns

    for column in REQUIRED_FEATURE_COLUMNS:
        if column not in lower_columns:
            result["missing_columns"].append(column)

    if require_target and "aqi_category" not in lower_columns:
        result["missing_columns"].append("aqi_category")

    if result["missing_columns"]:
        result["is_valid"] = False
        result["errors"].append("Missing required feature columns")
        return result

    numeric_columns = REQUIRED_FEATURE_COLUMNS + OPTIONAL_FEATURE_COLUMNS
    for column in numeric_columns:
        if column in check_df.columns:
            try:
                pd.to_numeric(check_df[column], errors="raise")
            except Exception:
                result["invalid_columns"].append(column)

    if result["invalid_columns"]:
        result["is_valid"] = False
        result["errors"].append("Invalid numeric values in feature columns")

    return result
