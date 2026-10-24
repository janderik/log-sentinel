import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.models.anomaly_detector import AnomalyDetector
from src.models.classifier import LogClassifier
from src.models.pattern_recognizer import PatternRecognizer


class TestAnomalyDetector:
    def test_fit_and_predict(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 10))
        X[:10] += 5
        df = pd.DataFrame(X)
        detector = AnomalyDetector(contamination=0.1)
        preds = detector.fit_predict(df)
        assert len(preds) == 200
        assert preds.dtype == np.int64

    def test_anomaly_score(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 5))
        df = pd.DataFrame(X)
        detector = AnomalyDetector(contamination=0.1)
        detector.fit(df)
        scores = detector.anomaly_score(df)
        assert scores.shape == (100,)
        assert 0.0 <= scores.min() <= scores.max() <= 1.0

    def test_save_load(self, tmp_path) -> None:
        df = pd.DataFrame(np.random.default_rng(42).normal(0, 1, (50, 5)))
        detector = AnomalyDetector()
        detector.fit(df)
        path = str(tmp_path / "anomaly.joblib")
        detector.save(path)
        loaded = AnomalyDetector.load(path)
        assert loaded.is_fitted
        preds = loaded.predict(df)
        assert len(preds) == 50

    def test_predict_without_fit_raises(self) -> None:
        detector = AnomalyDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            detector.predict(pd.DataFrame({"a": [1, 2, 3]}))


class TestLogClassifier:
    def test_fit_and_predict(self) -> None:
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5, n_classes=3, random_state=42
        )
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        labels = [["error", "warn", "info"][i % 3] for i in y]

        classifier = LogClassifier()
        classifier.fit(df, labels)
        preds = classifier.predict(df)
        assert len(preds) == 200
        assert all(p in ("error", "warn", "info") for p in preds)

    def test_train_with_eval(self) -> None:
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5, n_classes=2, random_state=42
        )
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        labels = [["error", "info"][i] for i in y]

        classifier = LogClassifier()
        results = classifier.train_with_eval(df, labels, test_size=0.3)
        assert "accuracy" in results
        assert "f1_weighted" in results
        assert 0 <= results["accuracy"] <= 1

    def test_save_load(self, tmp_path) -> None:
        X = np.random.default_rng(42).normal(0, 1, (50, 5))
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        labels = ["error" if i < 25 else "info" for i in range(50)]

        classifier = LogClassifier()
        classifier.fit(df, labels)
        path = str(tmp_path / "classifier.joblib")
        classifier.save(path)
        loaded = LogClassifier.load(path)
        assert loaded.is_fitted
        preds = loaded.predict(df)
        assert len(preds) == 50

    def test_predict_without_fit_raises(self) -> None:
        classifier = LogClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            classifier.predict(pd.DataFrame({"a": [1, 2, 3]}))


class TestPatternRecognizer:
    def test_fit_and_query(self) -> None:
        messages = [
            "Failed password for admin from 10.0.0.1",
            "Failed password for admin from 10.0.0.2",
            "Accepted publickey for frank from 192.168.1.1",
            "Error connecting to database: timeout",
            "Connection refused on port 22",
            "Successful login for user alice",
        ]
        recognizer = PatternRecognizer(max_features=100)
        recognizer.fit(messages)
        results = recognizer.query("Failed password from 10.0.0.3", top_k=3)
        assert len(results) <= 3
        if results:
            assert results[0]["similarity"] > 0

    def test_find_clusters(self) -> None:
        messages = [
            "Failed password for admin from 10.0.0.1",
            "Failed password for admin from 10.0.0.2",
            "Accepted publickey for frank from 192.168.1.1",
            "Failed password for root from 10.0.0.3",
            "Connection timeout on port 80",
        ]
        recognizer = PatternRecognizer(max_features=100)
        recognizer.fit(messages)
        clusters = recognizer.find_clusters(threshold=0.5)
        assert isinstance(clusters, list)

    def test_save_load(self, tmp_path) -> None:
        messages = ["test message one", "test message two"]
        recognizer = PatternRecognizer(max_features=100)
        recognizer.fit(messages)
        path = str(tmp_path / "pattern.joblib")
        recognizer.save(path)
        loaded = PatternRecognizer.load(path)
        assert loaded.is_fitted
        results = loaded.query("test message")
        assert len(results) > 0


def test_classifier_evaluate_on_subset_of_classes() -> None:
    """Evaluation data may lack some training classes; that must not crash."""
    X, y = make_classification(n_samples=150, n_features=6, n_informative=4, n_classes=3, random_state=0)
    names = np.array(["INFO", "WARN", "ERROR"])[y].tolist()
    clf = LogClassifier().fit(pd.DataFrame(X), names)
    only_two = [i for i, n in enumerate(names) if n != "ERROR"][:40]
    report = clf.evaluate(pd.DataFrame(X[only_two]), [names[i] for i in only_two])
    assert 0.0 <= report["accuracy"] <= 1.0
    assert "ERROR" in report["classification_report"]


def test_classifier_evaluate_rejects_unknown_labels() -> None:
    X, y = make_classification(n_samples=60, n_features=4, n_informative=3, n_redundant=0, n_classes=2, random_state=1)
    clf = LogClassifier().fit(pd.DataFrame(X), ["A" if v else "B" for v in y])
    with pytest.raises(ValueError, match="not seen during training"):
        clf.evaluate(pd.DataFrame(X[:3]), ["A", "B", "C"])


def test_anomaly_score_is_independent_of_the_batch() -> None:
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.normal(0, 1, (300, 5)))
    detector = AnomalyDetector().fit(train)
    normal, outlier = [0.0] * 5, [8.0] * 5

    alone = detector.anomaly_score(pd.DataFrame([outlier]))[0]
    in_batch = detector.anomaly_score(pd.DataFrame([normal, outlier]))[1]
    assert alone == pytest.approx(in_batch)
    # A single line no longer collapses to 0.5: outliers score clearly higher.
    assert detector.anomaly_score(pd.DataFrame([normal]))[0] < alone
    assert alone > 0.9


def test_anomaly_score_range_survives_save_load(tmp_path) -> None:
    train = pd.DataFrame(np.random.default_rng(1).normal(0, 1, (100, 4)))
    detector = AnomalyDetector().fit(train)
    path = str(tmp_path / "a.joblib")
    detector.save(path)
    sample = pd.DataFrame([[5.0] * 4])
    assert AnomalyDetector.load(path).anomaly_score(sample)[0] == pytest.approx(
        detector.anomaly_score(sample)[0]
    )
