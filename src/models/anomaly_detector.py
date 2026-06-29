from typing import Optional, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib


class AnomalyDetector:
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: str = "auto",
        random_state: int = 42,
    ) -> None:
        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False
        self._feature_names: Optional[list[str]] = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, features: pd.DataFrame) -> "AnomalyDetector":
        self._feature_names = features.columns.tolist()
        self._model.fit(features)
        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        raw = self._model.predict(features)
        return np.where(raw == -1, 1, 0)

    def score_samples(self, features: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.score_samples(features)

    def anomaly_score(self, features: pd.DataFrame) -> np.ndarray:
        scores = self.score_samples(features)
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s == 0:
            return np.full_like(scores, 0.5)
        normalized = 1.0 - (scores - min_s) / (max_s - min_s)
        return normalized

    def fit_predict(self, features: pd.DataFrame) -> np.ndarray:
        self.fit(features)
        return self.predict(features)

    def save(self, path: str) -> None:
        data = {"model": self._model, "feature_names": self._feature_names}
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str) -> "AnomalyDetector":
        data = joblib.load(path)
        inst = cls()
        inst._model = data["model"]
        inst._feature_names = data.get("feature_names")
        inst._fitted = True
        return inst
