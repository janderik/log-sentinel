from src.parser.log_parser import LogFormat, LogParser
from src.parser.normalizer import LogNormalizer


class TestLogParser:
    def setup_method(self) -> None:
        self.parser = LogParser()

    def test_detect_apache(self) -> None:
        line = '192.168.1.1 - frank [10/Oct/2023:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326'
        fmt = self.parser.detect_format(line)
        assert fmt == LogFormat.APACHE_COMBINED

    def test_parse_apache(self) -> None:
        line = '192.168.1.1 - frank [10/Oct/2023:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326'
        parsed = self.parser.parse_line(line)
        assert parsed is not None
        assert parsed.source == "192.168.1.1"
        assert parsed.extra["status"] == 200
        assert parsed.extra["method"] == "GET"
        assert parsed.extra["path"] == "/index.html"

    def test_parse_apache_error(self) -> None:
        line = '203.0.113.5 - - [10/Oct/2023:14:40:05 -0700] "POST /api/login HTTP/1.1" 500 255'
        parsed = self.parser.parse_line(line)
        assert parsed is not None
        assert parsed.severity == "ERROR"
        assert "server_error" in parsed.extra or "server_error" in parsed.__dict__

    def test_detect_syslog(self) -> None:
        line = "Oct 10 13:55:36 webserver01 sshd[1234]: Accepted publickey for frank from 192.168.1.1"
        fmt = self.parser.detect_format(line)
        assert fmt == LogFormat.SYSLOG_RFC3164

    def test_parse_syslog(self) -> None:
        line = "Oct 10 14:01:12 webserver01 sshd[1234]: Failed password for invalid user admin from 10.0.0.2"
        parsed = self.parser.parse_line(line)
        assert parsed is not None
        assert parsed.source == "webserver01"
        assert "failed password" in parsed.message.lower()
        assert parsed.extra["app_name"] == "sshd"

    def test_detect_windows(self) -> None:
        line = "2023-10-10T13:55:36.000+0000, Security, 4624, Logon succeeded"
        fmt = self.parser.detect_format(line)
        assert fmt == LogFormat.WINDOWS_EVENT

    def test_parse_windows(self) -> None:
        line = "2023-10-10T14:01:12.000+0000, Security, 4625, An account failed to log on., Failure"
        parsed = self.parser.parse_line(line)
        assert parsed is not None
        assert parsed.source == "Security"
        assert parsed.extra["event_id"] == 4625

    def test_parse_json(self) -> None:
        line = '{"timestamp":"2023-10-10T14:00:00Z","severity":"ERROR","message":"Something broke","source":"app1"}'
        parsed = self.parser.parse_line(line)
        assert parsed is not None
        assert parsed.severity == "ERROR"
        assert parsed.source == "app1"
        assert parsed.message == "Something broke"

    def test_unparseable_line(self) -> None:
        line = "this is not a valid log line at all"
        parsed = self.parser.parse_line(line)
        assert parsed is None

    def test_empty_line(self) -> None:
        assert self.parser.parse_line("") is None
        assert self.parser.parse_line("  ") is None


class TestLogNormalizer:
    def setup_method(self) -> None:
        self.parser = LogParser()
        self.normalizer = LogNormalizer()

    def test_normalize_apache(self) -> None:
        line = '192.168.1.1 - frank [10/Oct/2023:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326'
        parsed = self.parser.parse_line(line)
        assert parsed is not None
        normalized = self.normalizer.normalize(parsed)
        assert normalized.log_type == "http"
        assert normalized.severity == "INFO"
        assert normalized.source == "192.168.1.1"

    def test_normalize_syslog_error(self) -> None:
        line = "Oct 10 14:01:12 webserver01 sshd[1234]: Failed password for invalid user admin from 10.0.0.2"
        parsed = self.parser.parse_line(line)
        assert parsed is not None
        normalized = self.normalizer.normalize(parsed)
        assert "auth_failure" in normalized.tags
        assert "alert" in normalized.tags

    def test_normalize_windows(self) -> None:
        line = "2023-10-10T14:01:12.000+0000, Security, 4625, An account failed to log on., Failure"
        parsed = self.parser.parse_line(line)
        assert parsed is not None
        normalized = self.normalizer.normalize(parsed)
        assert normalized.log_type == "windows_event"

    def test_severity_score(self) -> None:
        assert self.normalizer.severity_score("ERROR") == 3
        assert self.normalizer.severity_score("WARN") == 2
        assert self.normalizer.severity_score("INFO") == 1
        assert self.normalizer.severity_score("DEBUG") == 0
