import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.features.extractor import FeatureExtractor
from src.models.anomaly_detector import AnomalyDetector
from src.models.classifier import LogClassifier
from src.models.pattern_recognizer import PatternRecognizer
from src.parser.log_parser import LogParser
from src.parser.normalizer import LogNormalizer

logger = logging.getLogger(__name__)
router = APIRouter()


class LogEntryIn(BaseModel):
    line: str
    format: str | None = None


class LogAnalysisOut(BaseModel):
    parsed: dict
    normalized: dict
    features: dict
    anomaly_score: float | None = None
    is_anomaly: bool | None = None
    predicted_severity: str | None = None
    similar_patterns: list[dict] = []


class BatchAnalysisIn(BaseModel):
    lines: list[str]
    format: str | None = None


class BatchAnalysisOut(BaseModel):
    results: list[LogAnalysisOut]
    anomaly_count: int = 0
    total_count: int = 0


@router.post("/analyze", response_model=LogAnalysisOut)
async def analyze_log(entry: LogEntryIn, request: Request):
    return _analyze(request.app.state, entry)


def _analyze(state, entry: LogEntryIn) -> LogAnalysisOut:
    parser: LogParser = state.parser
    normalizer: LogNormalizer = state.normalizer
    extractor: FeatureExtractor = state.feature_extractor
    anomaly_detector: AnomalyDetector | None = state.anomaly_detector
    classifier: LogClassifier | None = state.classifier
    pattern_recognizer: PatternRecognizer | None = state.pattern_recognizer

    fmt = entry.format
    if fmt and fmt not in ("apache_combined", "syslog_rfc3164", "windows_event", "json"):
        raise HTTPException(status_code=400, detail=f"Unknown format: {fmt}")

    parsed = parser.parse_line(entry.line, fmt)
    if parsed is None:
        raise HTTPException(status_code=422, detail="Unable to parse log line")

    normalized = normalizer.normalize(parsed)
    features = extractor.extract(normalized)

    anomaly_score: float | None = None
    is_anomaly: bool | None = None
    predicted_severity: str | None = None
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
async def analyze_batch(batch: BatchAnalysisIn, request: Request):
    results: list[LogAnalysisOut] = []
    anomaly_count = 0

    for line in batch.lines:
        try:
            result = _analyze(request.app.state, LogEntryIn(line=line, format=batch.format))
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
async def model_status(request: Request):
    state = request.app.state
    return {
        "anomaly_detector": state.anomaly_detector is not None,
        "classifier": state.classifier is not None,
        "pattern_recognizer": state.pattern_recognizer is not None,
    }
