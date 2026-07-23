import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

N_KMEANS_BACKGROUND = 50


class ShapExplainer:
    def __init__(self, model, background_data: pd.DataFrame):
        self.model = model
        self.background_data = background_data.copy()
        self.feature_names = list(background_data.columns)

        n_clusters = min(N_KMEANS_BACKGROUND, len(self.background_data))
        kmeans_obj = shap.kmeans(self.background_data, n_clusters)
        background_kmeans = pd.DataFrame(kmeans_obj.data, columns=self.feature_names)
        logger.info("SHAP background: %d kmeans clusters from %d samples", n_clusters, len(self.background_data))

        self.explainer = shap.KernelExplainer(
            self._model_predict, background_kmeans, link="identity"
        )

    def _model_predict(self, data: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(data, columns=self.feature_names)
        if hasattr(self.model, "predict_proba") and callable(getattr(self.model, "predict_proba", None)):
            return self.model.predict_proba(df)
        return self.model.decision_function(df)

    def get_summary(self):
        n_clusters = min(25, len(self.background_data))
        kmeans_obj = shap.kmeans(self.background_data, n_clusters)
        sample = pd.DataFrame(kmeans_obj.data, columns=self.feature_names)
        shap_values = self.explainer.shap_values(sample)

        if isinstance(shap_values, list):
            aggregated = np.mean([np.abs(values).mean(axis=0) for values in shap_values], axis=0)
        else:
            aggregated = np.mean(np.abs(shap_values), axis=0)

        return {
            feature: float(aggregated[idx])
            for idx, feature in enumerate(self.feature_names)
        }

    def explain_instance(self, input_df: pd.DataFrame):
        shap_values = self.explainer.shap_values(input_df)
        if isinstance(shap_values, list):
            values = [vals[0].tolist() if len(vals) else [] for vals in shap_values]
        else:
            values = shap_values[0].tolist() if shap_values.ndim > 1 else shap_values.tolist()

        ev = getattr(self.explainer, "expected_value", None)
        if ev is not None:
            base_values = [float(v) for v in ev] if hasattr(ev, "__iter__") else [float(ev)]
        else:
            base_values = []
        return {
            "feature_names": self.feature_names,
            "shap_values": values,
            "base_values": base_values,
        }

    def save_summary_plot(self, X: pd.DataFrame, save_dir: Path) -> Path:
        """Save SHAP beeswarm plot as high-res PNG for non-Python stakeholders."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_dir.mkdir(parents=True, exist_ok=True)
        shap_values = self.explainer.shap_values(X)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()

        png_path = save_dir / "shap_beeswarm.png"
        plt.savefig(str(png_path), dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("SHAP beeswarm plot saved to %s", png_path)
        return png_path
