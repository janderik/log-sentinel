from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


class LogClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int = 42,
    ) -> None:
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self._fitted = False
        self._feature_names: list[str] | None = None
        self._label_encoder: dict[str, int] | None = None
        self._reverse_encoder: dict[int, str] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, features: pd.DataFrame, labels: list[str]) -> "LogClassifier":
        unique = sorted(set(labels))
        self._label_encoder = {lab: i for i, lab in enumerate(unique)}
        self._reverse_encoder = {i: lab for lab, i in self._label_encoder.items()}
        y = np.array([self._label_encoder[l] for l in labels])
        self._feature_names = features.columns.tolist()
        self._model.fit(features, y)
        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> list[str]:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        raw = self._model.predict(features)
        return [self._reverse_encoder[int(i)] for i in raw]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict_proba(features)

    def evaluate(self, features: pd.DataFrame, labels: list[str]) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        preds = self.predict(features)
        y_true = np.array([self._label_encoder[l] for l in labels])
        y_pred = np.array([self._label_encoder[p] for p in preds])
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
            "classification_report": classification_report(
                y_true, y_pred, target_names=list(self._label_encoder.keys()), output_dict=True
            ),
        }

    def train_with_eval(
        self, features: pd.DataFrame, labels: list[str], test_size: float = 0.2, random_state: int = 42
    ) -> dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        self.fit(X_train, y_train)
        return self.evaluate(X_test, y_test)

    def save(self, path: str) -> None:
        data = {
            "model": self._model,
            "feature_names": self._feature_names,
            "label_encoder": self._label_encoder,
            "reverse_encoder": self._reverse_encoder,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str) -> "LogClassifier":
        data = joblib.load(path)
        inst = cls()
        inst._model = data["model"]
        inst._feature_names = data.get("feature_names")
        inst._label_encoder = data.get("label_encoder")
        inst._reverse_encoder = data.get("reverse_encoder")
        inst._fitted = True
        return inst
