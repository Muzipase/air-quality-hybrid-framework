import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VIF_THRESHOLD = 10.0


def compute_vif(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each feature.

    VIF measures how much the variance of a regression coefficient is inflated
    due to multicollinearity with other features. VIF > 10 indicates high
    multicollinearity that may degrade model interpretability.
    """
    if features is None:
        features = list(df.columns)

    if len(features) < 2:
        return pd.DataFrame({"feature": features, "vif": [1.0] * len(features)})

    X = df[features].copy()
    X = X.apply(pd.to_numeric, errors="coerce").dropna()

    vif_data = []
    for i, col in enumerate(features):
        y = X[col].values
        X_others = X.drop(columns=[col]).values

        if X_others.shape[1] == 0:
            vif_data.append({"feature": col, "vif": 1.0})
            continue

        X_with_const = np.column_stack([np.ones(len(X_others)), X_others])
        try:
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            y_pred = X_with_const @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            vif = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else float("inf")
        except np.linalg.LinAlgError:
            vif = float("inf")

        vif_data.append({"feature": col, "vif": round(vif, 2)})

    return pd.DataFrame(vif_data).sort_values("vif", ascending=False).reset_index(drop=True)


def flag_high_vif(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    threshold: float = VIF_THRESHOLD,
) -> Tuple[List[str], pd.DataFrame]:
    """Identify features exceeding the VIF threshold.

    Returns a list of features to consider dropping and the full VIF table.
    """
    vif_df = compute_vif(df, features)
    high_vif = vif_df[vif_df["vif"] > threshold]["feature"].tolist()

    if high_vif:
        logger.warning(
            "High multicollinearity detected (VIF > %.1f): %s",
            threshold,
            ", ".join(f"{f} ({vif_df[vif_df.feature == f]['vif'].values[0]})" for f in high_vif),
        )
    else:
        logger.info("No features exceed VIF threshold of %.1f", threshold)

    return high_vif, vif_df
