import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from re import Pattern


@dataclass
class ParsedLog:
    raw: str
    timestamp: str | None = None
    source: str | None = None
    severity: str | None = None
    message: str | None = None
    extra: dict = field(default_factory=dict)


class LogFormat:
    APACHE_COMBINED = "apache_combined"
    NGINX = "apache_combined"
    SYSLOG_RFC3164 = "syslog_rfc3164"
    WINDOWS_EVENT = "windows_event"
    JSON = "json"
    CUSTOM = "custom"


APACHE_PATTERN = re.compile(
    r'(?P<remote_addr>\S+)\s+'               # 192.168.1.1
    r'(?P<ident>\S+)\s+'                      # -
    r'(?P<remote_user>\S+)\s+'                # frank
    r'\[(?P<timestamp>[^\]]+)\]\s+'           # 10/Oct/2023:13:55:36 -0700
    r'"(?P<method>\S+)\s+'                     # GET
    r'(?P<path>\S+)\s+'                        # /index.html
    r'(?P<protocol>\S+)"\s+'                   # HTTP/1.1
    r'(?P<status>\d{3})\s+'                    # 200
    r'(?P<size>\S+)'                           # 2326
)

SYSLOG_PATTERN = re.compile(
    r'(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+'  # Oct 10 13:55:36
    r'(?P<hostname>\S+)\s+'                                      # webserver01
    r'(?P<app_name>[^\[:]+)'                                     # sshd or sshd[1234]
    r'(?:\[(?P<pid>\d+)\])?\s*:\s+'                              # [1234]:
    r'(?P<message>.+)'                                           # Accepted publickey...
)

WINDOWS_PATTERN = re.compile(
    r'(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{4}),\s*'
    r'(?P<log_name>[^,]+),\s*'
    r'(?P<event_id>\d+),\s*'
    r'(?P<message>.+)'
)


class LogParser:
    def __init__(self) -> None:
        self._formats: dict[str, Pattern] = {
            LogFormat.APACHE_COMBINED: APACHE_PATTERN,
            LogFormat.SYSLOG_RFC3164: SYSLOG_PATTERN,
            LogFormat.WINDOWS_EVENT: WINDOWS_PATTERN,
        }
        self._custom_patterns: dict[str, Pattern] = {}

    def register_format(self, name: str, pattern: str) -> None:
        self._custom_patterns[name] = re.compile(pattern)
        self._formats[LogFormat.CUSTOM] = self._custom_patterns[name]

    def detect_format(self, line: str) -> str | None:
        if line.lstrip().startswith("{"):
            return LogFormat.JSON
        for fmt_name, pat in self._formats.items():
            if pat.match(line):
                return fmt_name
        return None

    def parse_line(self, line: str, fmt: str | None = None) -> ParsedLog | None:
        line = line.strip()
        if not line:
            return None
        if fmt is None:
            fmt = self.detect_format(line)
            if fmt is None:
                return None
        if fmt == LogFormat.JSON:
            return self._parse_json(line)
        pat = self._formats.get(fmt)
        if pat is None:
            return None
        m = pat.match(line)
        if not m:
            return None
        return self._build_parsed_log(line, fmt, m)

    def parse_file(self, path: str, fmt: str | None = None) -> list[ParsedLog]:
        entries: list[ParsedLog] = []
        with Path(path).open(encoding="utf-8", errors="replace") as f:
            for line in f:
                entry = self.parse_line(line, fmt)
                if entry is not None:
                    entries.append(entry)
        return entries

    def parse_stream(self, lines: Iterator[str]) -> Iterator[ParsedLog]:
        for line in lines:
            entry = self.parse_line(line)
            if entry is not None:
                yield entry

    def _parse_json(self, line: str) -> ParsedLog | None:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        return ParsedLog(
            raw=line,
            timestamp=obj.get("timestamp") or obj.get("@timestamp") or obj.get("time"),
            source=obj.get("source") or obj.get("host") or obj.get("log_source"),
            severity=obj.get("severity") or obj.get("level") or obj.get("log_level"),
            message=obj.get("message") or obj.get("msg") or json.dumps(obj),
            extra={k: v for k, v in obj.items() if k not in ("timestamp", "source", "severity", "message", "level", "host")},
        )

    def _build_parsed_log(self, raw: str, fmt: str, m: re.Match) -> ParsedLog:
        if fmt == LogFormat.APACHE_COMBINED:
            return ParsedLog(
                raw=raw,
                timestamp=m.group("timestamp"),
                source=m.group("remote_addr"),
                severity=self._http_status_to_severity(int(m.group("status"))),
                message=f'{m.group("method")} {m.group("path")} -> {m.group("status")}',
                extra={
                    "method": m.group("method"),
                    "path": m.group("path"),
                    "protocol": m.group("protocol"),
                    "status": int(m.group("status")),
                    "size": m.group("size"),
                    "remote_user": m.group("remote_user"),
                    "ident": m.group("ident"),
                },
            )
        if fmt == LogFormat.SYSLOG_RFC3164:
            app = m.group("app_name").strip()
            pid = m.group("pid")
            return ParsedLog(
                raw=raw,
                timestamp=m.group("timestamp"),
                source=m.group("hostname"),
                severity=self._syslog_severity_from_message(m.group("message")),
                message=m.group("message"),
                extra={"app_name": app, "pid": int(pid) if pid else None, "hostname": m.group("hostname")},
            )
        if fmt == LogFormat.WINDOWS_EVENT:
            return ParsedLog(
                raw=raw,
                timestamp=m.group("timestamp"),
                source=m.group("log_name"),
                severity=self._windows_event_severity(int(m.group("event_id"))),
                message=m.group("message"),
                extra={"event_id": int(m.group("event_id"))},
            )
        return ParsedLog(raw=raw, message=raw)

    @staticmethod
    def _http_status_to_severity(status: int) -> str:
        if status >= 500:
            return "ERROR"
        if status >= 400:
            return "WARN"
        if status >= 300:
            return "INFO"
        return "INFO"

    @staticmethod
    def _syslog_severity_from_message(msg: str) -> str:
        lowered = msg.lower()
        if any(w in lowered for w in ("error", "failed", "failure", "critical")):
            return "ERROR"
        if any(w in lowered for w in ("warn", "warning")):
            return "WARN"
        return "INFO"

    @staticmethod
    def _windows_event_severity(event_id: int) -> str:
        if event_id in (4625, 4648, 1001):
            return "WARN"
        if event_id in (7045,):
            return "INFO"
        return "INFO"
