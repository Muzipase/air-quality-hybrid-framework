import logging
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from typing import Tuple

logger = logging.getLogger(__name__)

MAX_SYNTHETIC_RATIO = 1.5


def apply_smote_tomek(
    X: pd.DataFrame,
    y: pd.Series,
    max_synthetic_ratio: float = MAX_SYNTHETIC_RATIO,
) -> Tuple[pd.DataFrame, pd.Series]:
    if X is None or y is None:
        return pd.DataFrame(), pd.Series(dtype=object)

    min_class_count = y.value_counts().min()
    max_class_count = y.value_counts().max()
    k_neighbors = max(1, min(5, min_class_count - 1))

    synthetic_cap = int(max_class_count * max_synthetic_ratio)
    target_count = min(synthetic_cap, max_class_count)

    sampling_strategy = {}
    for cls, count in y.value_counts().items():
        if count < target_count:
            sampling_strategy[cls] = target_count

    if not sampling_strategy:
        logger.info("No oversampling needed — all classes already at or above cap.")
        return X.copy(), y.copy()

    logger.info("SMOTE cap: max_synthetic_ratio=%.1f, target=%d per minority class", max_synthetic_ratio, target_count)

    smote = SMOTE(
        random_state=42,
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
    )
    sampler = SMOTETomek(random_state=42, smote=smote)
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    X_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    y_balanced = pd.Series(y_resampled, name=y.name)

    return X_balanced, y_balanced
