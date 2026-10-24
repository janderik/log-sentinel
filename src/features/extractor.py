import math
import re
from datetime import datetime

import pandas as pd

from src.parser.normalizer import NormalizedLog


class FeatureExtractor:
    def __init__(self) -> None:
        self._ip_pattern = re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        )

    def extract(self, log: NormalizedLog) -> dict[str, float]:
        features: dict[str, float] = {}
        features.update(self._extract_temporal(log))
        features.update(self._extract_text_features(log))
        features.update(self._extract_categorical(log))
        features.update(self._extract_composite(log))
        return features

    def extract_batch(self, logs: list[NormalizedLog]) -> pd.DataFrame:
        records = [self.extract(log) for log in logs]
        df = pd.DataFrame(records)
        df = df.fillna(0.0)
        return df

    def _extract_temporal(self, log: NormalizedLog) -> dict[str, float]:
        features: dict[str, float] = {}
        ts = log.timestamp
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                features["hour"] = float(dt.hour)
                features["day_of_week"] = float(dt.weekday())
                features["day_of_month"] = float(dt.day)
                features["month"] = float(dt.month)
                features["minute"] = float(dt.minute)
                features["is_business_hours"] = 1.0 if 8 <= dt.hour < 18 else 0.0
                features["is_night"] = 1.0 if dt.hour < 6 or dt.hour >= 22 else 0.0
                features["is_weekend"] = 1.0 if dt.weekday() >= 5 else 0.0
            except (ValueError, TypeError):
                self._fill_temporal_defaults(features)
        else:
            self._fill_temporal_defaults(features)
        return features

    @staticmethod
    def _fill_temporal_defaults(features: dict[str, float]) -> None:
        features["hour"] = -1.0
        features["day_of_week"] = -1.0
        features["day_of_month"] = -1.0
        features["month"] = -1.0
        features["minute"] = -1.0
        features["is_business_hours"] = 0.0
        features["is_night"] = 0.0
        features["is_weekend"] = 0.0

    def _extract_text_features(self, log: NormalizedLog) -> dict[str, float]:
        msg = log.message or ""
        raw = log.raw or ""
        combined = f"{msg} {raw}"

        length = len(combined)
        word_count = len(combined.split())
        avg_word_length = length / max(word_count, 1)

        uppercase_ratio = sum(1 for c in combined if c.isupper()) / max(length, 1)
        digit_ratio = sum(1 for c in combined if c.isdigit()) / max(length, 1)
        space_ratio = sum(1 for c in combined if c.isspace()) / max(length, 1)
        special_char_ratio = sum(
            1 for c in combined if not c.isalnum() and not c.isspace()
        ) / max(length, 1)

        ip_count = len(self._ip_pattern.findall(combined))
        hex_seq_count = len(re.findall(r"0x[0-9a-fA-F]+", combined))
        uuid_count = len(
            re.findall(
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                combined,
            )
        )
        path_segments = len(re.findall(r"/\w+", combined))
        equal_signs = combined.count("=")
        colon_count = combined.count(":")

        entropy = self._shannon_entropy(combined)
        num_count = len(re.findall(r"\b\d+\b", combined))

        return {
            "length": float(length),
            "word_count": float(word_count),
            "avg_word_length": float(avg_word_length),
            "uppercase_ratio": float(uppercase_ratio),
            "digit_ratio": float(digit_ratio),
            "space_ratio": float(space_ratio),
            "special_char_ratio": float(special_char_ratio),
            "ip_count": float(ip_count),
            "hex_seq_count": float(hex_seq_count),
            "uuid_count": float(uuid_count),
            "path_segments": float(path_segments),
            "equal_signs": float(equal_signs),
            "colon_count": float(colon_count),
            "entropy": float(entropy),
            "num_count": float(num_count),
        }

    @staticmethod
    def _extract_categorical(log: NormalizedLog) -> dict[str, float]:
        severity_map = {"DEBUG": 0.0, "INFO": 1.0, "WARN": 2.0, "ERROR": 3.0, "CRITICAL": 4.0}
        type_map = {"http": 0.0, "syslog": 1.0, "windows_event": 2.0, "generic": 3.0, "unknown": 4.0}

        severity_score = severity_map.get(log.severity.upper(), 1.0)
        type_score = type_map.get(log.log_type, 4.0)
        is_alert = 1.0 if "alert" in log.tags else 0.0
        tag_count = float(len(log.tags))

        return {
            "severity_score": severity_score,
            "log_type_score": type_score,
            "is_alert": is_alert,
            "tag_count": tag_count,
        }

    @staticmethod
    def _extract_composite(log: NormalizedLog) -> dict[str, float]:
        msg = log.message or ""
        combined = f"{msg} {log.raw}"

        has_error_keyword = 1.0 if re.search(
            r"\b(error|failed|failure|exception|crash|timeout)\b", combined, re.IGNORECASE
        ) else 0.0
        has_warn_keyword = 1.0 if re.search(
            r"\b(warn|warning|caution)\b", combined, re.IGNORECASE
        ) else 0.0
        has_auth_keyword = 1.0 if re.search(
            r"\b(auth|login|password|credential|session)\b", combined, re.IGNORECASE
        ) else 0.0
        has_network_keyword = 1.0 if re.search(
            r"\b(network|connect|timeout|refused|reset)\b", combined, re.IGNORECASE
        ) else 0.0

        return {
            "has_error_keyword": has_error_keyword,
            "has_warn_keyword": has_warn_keyword,
            "has_auth_keyword": has_auth_keyword,
            "has_network_keyword": has_network_keyword,
        }

    @staticmethod
    def _shannon_entropy(text: str) -> float:
        if not text:
            return 0.0
        text = text.strip()
        if not text:
            return 0.0
        prob: dict[str, float] = {}
        for ch in text:
            prob[ch] = prob.get(ch, 0.0) + 1.0
        length = len(text)
        entropy = -sum(p / length * math.log2(p / length) for p in prob.values())
        return entropy
