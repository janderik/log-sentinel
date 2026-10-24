import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from src.features.extractor import FeatureExtractor
from src.parser.log_parser import LogParser
from src.parser.normalizer import LogNormalizer

from .routes import router

logger = logging.getLogger(__name__)


def _load_models(app: FastAPI, model_dir: str) -> None:
    """Load trained models if present; the API also works without them."""
    model_path = Path(model_dir)
    if not model_path.exists():
        logger.warning("Model directory %s not found; ML predictions disabled", model_dir)
        return
    from src.models.anomaly_detector import AnomalyDetector
    from src.models.classifier import LogClassifier
    from src.models.pattern_recognizer import PatternRecognizer

    anomaly_path = model_path / "anomaly_detector.joblib"
    classifier_path = model_path / "classifier.joblib"
    pattern_path = model_path / "pattern_recognizer.joblib"

    if anomaly_path.exists():
        app.state.anomaly_detector = AnomalyDetector.load(str(anomaly_path))
        logger.info("Loaded anomaly detector from %s", anomaly_path)
    if classifier_path.exists():
        app.state.classifier = LogClassifier.load(str(classifier_path))
        logger.info("Loaded classifier from %s", classifier_path)
    if pattern_path.exists():
        app.state.pattern_recognizer = PatternRecognizer.load(str(pattern_path))
        logger.info("Loaded pattern recognizer from %s", pattern_path)


def create_app(model_dir: str = "models") -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _load_models(app, model_dir)
        yield

    app = FastAPI(
        title="LogSentinel API",
        description="Real-time log analysis with ML-powered anomaly detection, "
                    "classification, and pattern recognition",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.model_dir = model_dir
    # Parsing and feature extraction need no trained model: always available.
    app.state.parser = LogParser()
    app.state.normalizer = LogNormalizer()
    app.state.feature_extractor = FeatureExtractor()
    app.state.anomaly_detector = None
    app.state.classifier = None
    app.state.pattern_recognizer = None

    app.include_router(router)

    @app.get("/health")
    async def health_check():
        return {
            "status": "ok",
            "anomaly_detector": app.state.anomaly_detector is not None,
            "classifier": app.state.classifier is not None,
            "pattern_recognizer": app.state.pattern_recognizer is not None,
        }

    return app
