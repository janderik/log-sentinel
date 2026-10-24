from rich.console import Console

from src.visualization.dashboard import LogDashboard


def test_dashboard_renders_all_panels() -> None:
    dashboard = LogDashboard()
    dashboard.ingest('203.0.113.5 - - [10/Oct/2023:14:40:05 -0700] "GET / HTTP/1.1" 503 0')
    console = Console(record=True, width=120, height=40)
    console.print(dashboard.render())
    text = console.export_text()
    assert "Total Logs Processed" in text
    assert "Severity" in text and "Log Type" in text  # the split row panels
    assert "Recent Anomalies" in text
