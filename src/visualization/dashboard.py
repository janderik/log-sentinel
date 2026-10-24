import time
from collections import Counter, deque

import pandas as pd
from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.text import Text

from src.features.extractor import FeatureExtractor
from src.models.anomaly_detector import AnomalyDetector
from src.parser.log_parser import LogParser
from src.parser.normalizer import LogNormalizer


class LogDashboard:
    def __init__(
        self,
        anomaly_detector: AnomalyDetector | None = None,
        max_history: int = 100,
    ) -> None:
        self.anomaly_detector = anomaly_detector
        self.parser = LogParser()
        self.normalizer = LogNormalizer()
        self.extractor = FeatureExtractor()

        self.history: deque = deque(maxlen=max_history)
        self.anomaly_history: deque = deque(maxlen=max_history)
        self.severity_counter: Counter = Counter()
        self.type_counter: Counter = Counter()
        self.total_count = 0
        self.anomaly_count = 0
        self.start_time = time.time()

    def ingest(self, line: str) -> None:
        parsed = self.parser.parse_line(line)
        if parsed is None:
            return
        normalized = self.normalizer.normalize(parsed)
        features = self.extractor.extract(normalized)

        is_anomaly = False
        anomaly_score = 0.0

        if self.anomaly_detector is not None:
            feat_df = pd.DataFrame([features])
            scores = self.anomaly_detector.anomaly_score(feat_df)
            anomaly_score = float(scores[0])
            pred = self.anomaly_detector.predict(feat_df)
            is_anomaly = bool(pred[0] == 1)

        self.history.append(normalized)
        self.total_count += 1
        self.severity_counter[normalized.severity] += 1
        self.type_counter[normalized.log_type] += 1

        if is_anomaly:
            self.anomaly_count += 1
            self.anomaly_history.append({
                "message": normalized.message[:80],
                "score": anomaly_score,
                "severity": normalized.severity,
                "source": normalized.source,
            })

    def _build_stats_table(self) -> Table:
        table = Table(title="LogSentinel Dashboard", box=box.ROUNDED, title_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        elapsed = time.time() - self.start_time
        rate = self.total_count / max(elapsed, 1)

        table.add_row("Total Logs Processed", str(self.total_count))
        table.add_row("Anomalies Detected", str(self.anomaly_count))
        table.add_row("Rate (logs/sec)", f"{rate:.2f}")
        table.add_row("Uptime", f"{elapsed:.0f}s")
        table.add_row("Anomaly %", f"{(self.anomaly_count / max(self.total_count, 1)) * 100:.1f}%")
        return table

    def _build_severity_table(self) -> Table:
        table = Table(title="Severity Distribution", box=box.SIMPLE)
        table.add_column("Level", style="bold")
        table.add_column("Count", style="white")

        for sev in ("ERROR", "WARN", "INFO", "DEBUG", "CRITICAL"):
            count = self.severity_counter.get(sev, 0)
            style = "red" if sev in ("ERROR", "CRITICAL") else "yellow" if sev == "WARN" else "green"
            table.add_row(Text(sev, style=style), str(count))
        return table

    def _build_type_table(self) -> Table:
        table = Table(title="Log Type Distribution", box=box.SIMPLE)
        table.add_column("Type", style="bold")
        table.add_column("Count", style="white")

        for typ, count in self.type_counter.most_common():
            table.add_row(typ, str(count))
        return table

    def _build_anomaly_table(self) -> Table:
        table = Table(title="Recent Anomalies", box=box.HEAVY_HEAD, title_style="bold red")
        table.add_column("Score", style="red")
        table.add_column("Severity", style="bold")
        table.add_column("Message", style="white", no_wrap=True)

        for entry in list(self.anomaly_history)[-10:]:
            score_str = f"{entry['score']:.3f}"
            sev = entry["severity"]
            sev_style = "red" if sev == "ERROR" else "yellow"
            msg = entry["message"][:60]
            table.add_row(score_str, Text(sev, style=sev_style), msg)
        return table

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._build_stats_table(), name="stats"),
            Layout(
                Layout.split_row(
                    Layout(self._build_severity_table(), name="severity"),
                    Layout(self._build_type_table(), name="types"),
                ),
            ),
            Layout(self._build_anomaly_table(), name="anomalies"),
        )
        return layout

    def run_live(self, log_source: str) -> None:
        from pathlib import Path

        with Live(refresh_per_second=2, screen=True) as live:
            if log_source == "stdin":
                import sys
                for line in sys.stdin:
                    self.ingest(line)
                    live.update(self.render())
            else:
                path = Path(log_source)
                with path.open(encoding="utf-8", errors="replace") as f:
                    for line in f:
                        self.ingest(line)
                        live.update(self.render())
                        time.sleep(0.05)
