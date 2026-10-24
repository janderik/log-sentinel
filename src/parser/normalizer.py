from dataclasses import dataclass, field
from datetime import datetime

from .log_parser import ParsedLog


@dataclass
class NormalizedLog:
    timestamp: str | None = None
    source: str | None = None
    severity: str = "INFO"
    message: str = ""
    log_type: str = "unknown"
    tags: list[str] = field(default_factory=list)
    raw: str = ""
    extra: dict = field(default_factory=dict)


_SEVERITY_ORDER = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "CRITICAL": 4}
_SEVERITY_LABELS = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]


class LogNormalizer:
    def normalize(self, parsed: ParsedLog) -> NormalizedLog:
        tags: list[str] = []
        severity = (parsed.severity or "INFO").upper()
        if severity not in _SEVERITY_ORDER:
            severity = "INFO"
        log_type = self._infer_log_type(parsed)
        tags.extend(self._generate_tags(parsed, severity))

        return NormalizedLog(
            timestamp=self._normalize_timestamp(parsed.timestamp, log_type),
            source=parsed.source or "unknown",
            severity=severity,
            message=parsed.message or parsed.raw,
            log_type=log_type,
            tags=tags,
            raw=parsed.raw,
            extra=parsed.extra,
        )

    def normalize_batch(self, entries: list[ParsedLog]) -> list[NormalizedLog]:
        return [self.normalize(e) for e in entries]

    @staticmethod
    def severity_score(severity: str) -> int:
        return _SEVERITY_ORDER.get(severity.upper(), 1)

    @staticmethod
    def _infer_log_type(parsed: ParsedLog) -> str:
        extra = parsed.extra
        if "method" in extra and "path" in extra:
            return "http"
        if "app_name" in extra:
            return "syslog"
        if "event_id" in extra:
            return "windows_event"
        if "log_name" in extra:
            return "windows_event"
        return "generic"

    @staticmethod
    def _generate_tags(parsed: ParsedLog, severity: str) -> list[str]:
        tags: list[str] = []
        if severity in ("WARN", "ERROR", "CRITICAL"):
            tags.append("alert")
        extra = parsed.extra
        if "status" in extra:
            status = extra["status"]
            if isinstance(status, int):
                if status >= 400:
                    tags.append("http_error")
                if status == 401:
                    tags.append("unauthorized")
                if status == 403:
                    tags.append("forbidden")
                if status == 500:
                    tags.append("server_error")
                if status == 304:
                    tags.append("not_modified")
        if parsed.message:
            lowered = parsed.message.lower()
            if "failed password" in lowered:
                tags.append("auth_failure")
            if "accepted" in lowered:
                tags.append("auth_success")
            if "ufw block" in lowered:
                tags.append("firewall_block")
            if "bugcheck" in lowered or "rebooted" in lowered:
                tags.append("system_crash")
        if "ssh" in (parsed.message or "").lower():
            tags.append("ssh")
        return tags

    @staticmethod
    def _normalize_timestamp(ts: str | None, log_type: str) -> str | None:
        if ts is None:
            return None
        try:
            if log_type == "http":
                dt = datetime.strptime(ts, "%d/%b/%Y:%H:%M:%S %z")
            elif log_type == "syslog" or log_type == "generic":
                dt = datetime.strptime(ts, "%b %d %H:%M:%S")
                dt = dt.replace(year=datetime.now().year)
            elif log_type == "windows_event":
                dt = datetime.fromisoformat(ts)
            else:
                return ts
            return dt.isoformat()
        except (ValueError, TypeError):
            return ts
