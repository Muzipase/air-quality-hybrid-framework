"""
End-to-end ML pipeline: fetch data, preprocess, balance, train, evaluate.
Usage: python scripts/run_pipeline.py [--source auto|openmeteo|openaq] [--city Lusaka]
"""
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    BASELINE_MODEL_PATH, OPTIMIZED_MODEL_PATH,
    METRICS_PATH, SCALER_PATH, SHAP_PLOTS_DIR, ensure_dirs,
)
from src.ingestion.fetch_data import fetch_data
from src.preprocessing.clean_data import clean_data
from src.preprocessing.missing_values import fill_missing_values
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.scaling import fit_scaler, save_scaler, apply_scaler_to_dataframe
from src.preprocessing.split_dataset import split_data
from src.preprocessing.multicollinearity import compute_vif
from src.balancing.smote_tomek import apply_smote_tomek
from src.models.baseline_svm import train_baseline_svm
from src.models.optimized_svm import train_optimized_svm
from src.evaluation.metrics import compute_metrics

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(source: str = "auto", city: str = None):
    ensure_dirs()

    # 1. Fetch data
    logger.info("Fetching data (source=%s, city=%s)...", source, city)
    data = fetch_data(source=source, city=city)
    if data is None or data.empty:
        logger.error("No data retrieved. Aborting.")
        return False
    data.to_csv(RAW_DATA_PATH, index=False)
    logger.info("Fetched %d records, saved to %s", len(data), RAW_DATA_PATH)

    # 2. Preprocess
    logger.info("Preprocessing...")
    data = clean_data(data)
    data = fill_missing_values(data)
    data = engineer_features(data)

    feature_columns = [col for col in data.columns if col not in ['aqi_category', 'timestamp']]
    _, scaler = fit_scaler(data[feature_columns])
    save_scaler(scaler, SCALER_PATH)
    data = apply_scaler_to_dataframe(data, scaler, feature_columns)
    data.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info("Preprocessed %d records, saved to %s", len(data), PROCESSED_DATA_PATH)

    # 3. Multicollinearity check
    excluded = {"aqi_category", "timestamp", "location", "city", "country"}
    feature_cols = [c for c in data.columns if c not in excluded and pd.api.types.is_numeric_dtype(data[c])]
    vif_df = compute_vif(data, feature_cols)
    logger.info("VIF analysis:\n%s", vif_df.to_string(index=False))

    # 4. Split & balance
    X = data[feature_cols]
    y = data['aqi_category']

    X_train, X_test, y_train, y_test = split_data(X, y)
    logger.info("Split: train=%d, test=%d", len(X_train), len(X_test))

    X_train_bal, y_train_bal = apply_smote_tomek(X_train, y_train)
    logger.info("After SMOTE-Tomek: %d records (was %d)", len(X_train_bal), len(X_train))

    # 5. Train baseline
    logger.info("Training baseline SVM...")
    baseline_model = train_baseline_svm(X_train_bal, y_train_bal)
    joblib.dump(baseline_model, BASELINE_MODEL_PATH)
    logger.info("Baseline model saved to %s", BASELINE_MODEL_PATH)

    # 6. Train optimized
    logger.info("Training optimized SVM with Bayesian optimization...")
    optimized_model, best_params, study = train_optimized_svm(X_train_bal, y_train_bal)
    joblib.dump(optimized_model, OPTIMIZED_MODEL_PATH)
    logger.info("Optimized model saved to %s (best params: %s)", OPTIMIZED_MODEL_PATH, best_params)

    # 7. Evaluate
    y_pred = optimized_model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, save_path=METRICS_PATH)
    logger.info("Evaluation — Accuracy: %.4f, F1: %.4f", metrics['accuracy'], metrics['f1_score'])

    # 8. Save SHAP beeswarm plot
    try:
        from src.explainability.shap_explainer import ShapExplainer
        explainer = ShapExplainer(optimized_model, X_train_bal)
        explainer.save_summary_plot(X_test, SHAP_PLOTS_DIR)
    except Exception as e:
        logger.warning("Could not save SHAP plot: %s", e)

    logger.info("Pipeline complete.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Air Quality ML Pipeline")
    parser.add_argument("--source", default="auto", choices=["auto", "openmeteo", "openaq"])
    parser.add_argument("--city", default=None)
    args = parser.parse_args()
    success = run_pipeline(source=args.source, city=args.city)
    sys.exit(0 if success else 1)
