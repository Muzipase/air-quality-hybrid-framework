import pandas as pd
from sklearn.impute import SimpleImputer


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric features using median imputation."""
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()
    numeric_columns = [
        column
        for column in ["pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity", "wind_speed", "rainfall"]
        if column in data.columns
    ]

    if numeric_columns:
        columns_with_observed_values = [column for column in numeric_columns if data[column].notna().any()]
        columns_without_observed_values = [column for column in numeric_columns if column not in columns_with_observed_values]

        if columns_with_observed_values:
            imputer = SimpleImputer(strategy="median")
            imputed_values = pd.DataFrame(
                imputer.fit_transform(data[columns_with_observed_values]),
                columns=columns_with_observed_values,
                index=data.index,
            )
            data.loc[:, columns_with_observed_values] = imputed_values

        if columns_without_observed_values:
            data.loc[:, columns_without_observed_values] = 0.0

    if "aqi_category" in data.columns:
        data["aqi_category"] = data["aqi_category"].ffill().fillna("Moderate")

    return data
