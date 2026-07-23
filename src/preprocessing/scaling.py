import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple


def fit_scaler(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Fit a scaler on numeric columns and return scaled data with the trained scaler."""
    if df is None or df.empty:
        return df, StandardScaler()

    numeric_df = df.select_dtypes(include=["number"]).copy()
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(numeric_df)
    return pd.DataFrame(scaled_values, columns=numeric_df.columns, index=numeric_df.index), scaler


def apply_scaler_to_dataframe(df: pd.DataFrame, scaler: StandardScaler, feature_columns: Optional[list] = None) -> pd.DataFrame:
    """Apply a fitted scaler to numeric feature columns without breaking non-numeric columns."""
    if df is None or df.empty or scaler is None:
        return df

    result = df.copy()
    columns = feature_columns or [col for col in result.columns if col not in ["aqi_category", "timestamp"]]
    numeric_columns = list(getattr(scaler, "feature_names_in_", []))

    if not numeric_columns:
        numeric_columns = [
            col for col in columns
            if col in result.columns
            and pd.api.types.is_numeric_dtype(result[col])
            and not pd.api.types.is_bool_dtype(result[col])
        ]

    if not numeric_columns:
        return result

    for col in numeric_columns:
        if col in result.columns:
            result[col] = result[col].astype(float)

    scaled_values = scaler.transform(result[numeric_columns])
    result.loc[:, numeric_columns] = scaled_values
    return result


def transform_with_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    numeric_df = df.select_dtypes(include=["number"]).copy()
    scaled_values = scaler.transform(numeric_df)
    return pd.DataFrame(scaled_values, columns=numeric_df.columns, index=numeric_df.index)


def save_scaler(scaler: StandardScaler, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path) -> Optional[StandardScaler]:
    if not path.exists():
        return None
    return joblib.load(path)
