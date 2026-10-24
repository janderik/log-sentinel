import pandas as pd

from src.features.extractor import FeatureExtractor
from src.features.vectorizer import LogVectorizer
from src.parser.log_parser import LogParser
from src.parser.normalizer import LogNormalizer


class TestFeatureExtractor:
    def setup_method(self) -> None:
        self.parser = LogParser()
        self.normalizer = LogNormalizer()
        self.extractor = FeatureExtractor()

    def _normalize(self, line: str):
        parsed = self.parser.parse_line(line)
        assert parsed is not None
        return self.normalizer.normalize(parsed)

    def test_extract_apache_log(self) -> None:
        line = '192.168.1.1 - frank [10/Oct/2023:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326'
        norm = self._normalize(line)
        features = self.extractor.extract(norm)
        assert "hour" in features
        assert "length" in features
        assert "entropy" in features
        assert "special_char_ratio" in features
        assert "severity_score" in features
        assert "log_type_score" in features

    def test_extract_batch(self) -> None:
        lines = [
            '192.168.1.1 - frank [10/Oct/2023:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326',
            '10.0.0.2 - - [10/Oct/2023:14:01:12 -0700] "POST /api/login HTTP/1.1" 401 128',
        ]
        normalized = [self._normalize(l) for l in lines]
        df = self.extractor.extract_batch(normalized)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 2
        assert df.shape[1] > 20

    def test_text_features(self) -> None:
        line = "Oct 10 14:01:12 webserver01 sshd[1234]: Failed password for invalid user admin from 10.0.0.2"
        norm = self._normalize(line)
        features = self.extractor.extract(norm)
        assert features["ip_count"] >= 1
        assert features["word_count"] > 0
        assert features["entropy"] > 0

    def test_entropy(self) -> None:
        assert self.extractor._shannon_entropy("") == 0.0
        assert self.extractor._shannon_entropy("aaaa") > 0
        entropy_random = self.extractor._shannon_entropy("aB3$xY9#qR")
        entropy_repeated = self.extractor._shannon_entropy("aaaaaaaaaa")
        assert entropy_random > entropy_repeated

    def test_categorical_features(self) -> None:
        line = '203.0.113.5 - - [10/Oct/2023:14:40:05 -0700] "POST /api/login HTTP/1.1" 500 255'
        norm = self._normalize(line)
        features = self.extractor.extract(norm)
        assert "has_error_keyword" in features
        assert "is_alert" in features


class TestLogVectorizer:
    def setup_method(self) -> None:
        self.vectorizer = LogVectorizer(max_features=100)

    def test_fit_transform(self) -> None:
        texts = ["hello world", "error occurred", "failed login attempt", "system reboot"]
        result = self.vectorizer.fit_transform(texts)
        assert result.shape[0] == 4

    def test_save_load(self, tmp_path) -> None:
        texts = ["test log message one", "test log message two"]
        self.vectorizer.fit(texts)
        path = str(tmp_path / "vectorizer.joblib")
        self.vectorizer.save(path)
        loaded = LogVectorizer.load(path)
        assert loaded.is_fitted
        result = loaded.transform(["new test message"])
        assert result.shape[0] == 1
