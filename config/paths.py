from pathlib import Path

# Project root (two levels up from this file)
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
CONFIG_DIR = ROOT / "config"

RAW_DATA_PATH = DATA_DIR / "raw_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_svm.pkl"
OPTIMIZED_MODEL_PATH = MODELS_DIR / "optimized_svm.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
SHAP_VALUES_PATH = ARTIFACTS_DIR / "shap_values.json"
SHAP_PLOTS_DIR = ARTIFACTS_DIR / "shap_plots"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

def ensure_dirs():
    for p in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR):
        p.mkdir(parents=True, exist_ok=True)

__all__ = [
    "ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "ARTIFACTS_DIR",
    "CONFIG_DIR",
    "RAW_DATA_PATH",
    "PROCESSED_DATA_PATH",
    "BASELINE_MODEL_PATH",
    "OPTIMIZED_MODEL_PATH",
    "SCALER_PATH",
    "SHAP_VALUES_PATH",
    "SHAP_PLOTS_DIR",
    "METRICS_PATH",
    "ensure_dirs",
]

