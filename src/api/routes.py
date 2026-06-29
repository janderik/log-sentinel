import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import pandas as pd

from src.parser.log_parser import LogParser, LogFormat
from src.parser.normalizer import LogNormalizer
from src.features.extractor import FeatureExtractor
from src.models.anomaly_detector import AnomalyDetector
from src.models.classifier import LogClassifier
from src.models.pattern_recognizer import PatternRecognizer

logger = logging.getLogger(__name__)
router = APIRouter()


class LogEntryIn(BaseModel):
    line: str
    format: Optional[str] = None


class LogAnalysisOut(BaseModel):
    parsed: dict
    normalized: dict
    features: dict
    anomaly_score: Optional[float] = None
    is_anomaly: Optional[bool] = None
    predicted_severity: Optional[str] = None
    similar_patterns: list[dict] = []


class BatchAnalysisIn(BaseModel):
    lines: list[str]
    format: Optional[str] = None


class BatchAnalysisOut(BaseModel):
    results: list[LogAnalysisOut]
    anomaly_count: int = 0
    total_count: int = 0


@router.post("/analyze", response_model=LogAnalysisOut)
async def analyze_log(entry: LogEntryIn):
    app = router._app_ref
    parser: LogParser = app.state.parser
    normalizer: LogNormalizer = app.state.normalizer
    extractor: FeatureExtractor = app.state.feature_extractor
    anomaly_detector: Optional[AnomalyDetector] = app.state.anomaly_detector
    classifier: Optional[LogClassifier] = app.state.classifier
    pattern_recognizer: Optional[PatternRecognizer] = app.state.pattern_recognizer

    fmt = entry.format
    if fmt and fmt not in ("apache_combined", "syslog_rfc3164", "windows_event", "json"):
        raise HTTPException(status_code=400, detail=f"Unknown format: {fmt}")

    parsed = parser.parse_line(entry.line, fmt)
    if parsed is None:
        raise HTTPException(status_code=422, detail="Unable to parse log line")

    normalized = normalizer.normalize(parsed)
    features = extractor.extract(normalized)

    anomaly_score: Optional[float] = None
    is_anomaly: Optional[bool] = None
    predicted_severity: Optional[str] = None
    similar_patterns: list[dict] = []

    if anomaly_detector is not None:
        feat_df = pd.DataFrame([features])
        score = anomaly_detector.anomaly_score(feat_df)
        anomaly_score = float(round(score[0], 4))
        pred = anomaly_detector.predict(feat_df)
        is_anomaly = bool(pred[0] == 1)

    if classifier is not None:
        feat_df = pd.DataFrame([features])
        preds = classifier.predict(feat_df)
        predicted_severity = preds[0]

    if pattern_recognizer is not None:
        similar_patterns = pattern_recognizer.query(normalized.message, top_k=5)

    return LogAnalysisOut(
        parsed=parsed.__dict__,
        normalized=normalized.__dict__,
        features=features,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        predicted_severity=predicted_severity,
        similar_patterns=similar_patterns,
    )


@router.post("/analyze/batch", response_model=BatchAnalysisOut)
async def analyze_batch(batch: BatchAnalysisIn):
    app = router._app_ref
    results: list[LogAnalysisOut] = []
    anomaly_count = 0

    for line in batch.lines:
        try:
            result = await analyze_log(LogEntryIn(line=line, format=batch.format))
            results.append(result)
            if result.is_anomaly:
                anomaly_count += 1
        except HTTPException:
            results.append(LogAnalysisOut(
                parsed={"raw": line},
                normalized={"message": line, "severity": "UNKNOWN"},
                features={},
            ))

    return BatchAnalysisOut(
        results=results,
        anomaly_count=anomaly_count,
        total_count=len(batch.lines),
    )


@router.get("/models/status")
async def model_status():
    app = router._app_ref
    return {
        "anomaly_detector": app.state.anomaly_detector is not None,
        "classifier": app.state.classifier is not None,
        "pattern_recognizer": app.state.pattern_recognizer is not None,
    }
