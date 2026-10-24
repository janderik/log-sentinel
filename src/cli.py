import logging
import sys
import time
from pathlib import Path

import click
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from src.features.extractor import FeatureExtractor
from src.models.anomaly_detector import AnomalyDetector
from src.models.classifier import LogClassifier
from src.models.pattern_recognizer import PatternRecognizer
from src.parser.log_parser import LogParser
from src.parser.normalizer import LogNormalizer
from src.visualization.dashboard import LogDashboard

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", is_flag=True, help="Enable debug logging")
@click.option("--model-dir", default="models", help="Model storage directory")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, model_dir: str) -> None:
    """LogSentinel - ML-Powered Log Analyzer."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["model_dir"] = model_dir
    ctx.obj["verbose"] = verbose


def _load_or_create_models(model_dir: str) -> tuple[AnomalyDetector, LogClassifier, PatternRecognizer]:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    anomaly = AnomalyDetector()
    classifier = LogClassifier()
    recognizer = PatternRecognizer()

    anomaly_file = model_path / "anomaly_detector.joblib"
    classifier_file = model_path / "classifier.joblib"
    pattern_file = model_path / "pattern_recognizer.joblib"

    if anomaly_file.exists():
        anomaly = AnomalyDetector.load(str(anomaly_file))
        console.print("[green]Loaded existing anomaly detector[/]")
    if classifier_file.exists():
        classifier = LogClassifier.load(str(classifier_file))
        console.print("[green]Loaded existing classifier[/]")
    if pattern_file.exists():
        recognizer = PatternRecognizer.load(str(pattern_file))
        console.print("[green]Loaded existing pattern recognizer[/]")

    return anomaly, classifier, recognizer


@cli.command()
@click.option("--input-dir", "-i", required=True, help="Directory containing log files")
@click.option("--model-dir", "-m", default=None, help="Directory to save trained models")
@click.pass_context
def train(ctx: click.Context, input_dir: str, model_dir: str | None) -> None:
    """Train ML models on log data."""

    model_dir = model_dir or ctx.obj["model_dir"]
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_dir)
    if not input_path.exists():
        console.print(f"[red]Input directory not found: {input_dir}[/]")
        sys.exit(1)

    console.print(f"[cyan]Loading logs from {input_dir}...[/]")
    parser = LogParser()
    normalizer = LogNormalizer()
    extractor = FeatureExtractor()

    all_normalized = []
    for file_path in input_path.iterdir():
        if file_path.suffix in (".log", ".txt", ".json"):
            console.print(f"  Parsing {file_path.name}...")
            parsed = parser.parse_file(str(file_path))
            normalized = normalizer.normalize_batch(parsed)
            all_normalized.extend(normalized)

    if not all_normalized:
        console.print("[red]No log entries found.[/]")
        sys.exit(1)

    console.print(f"[green]Parsed {len(all_normalized)} log entries.[/]")

    console.print("[cyan]Extracting features...[/]")
    feature_df = extractor.extract_batch(all_normalized)
    messages = [n.message for n in all_normalized]
    labels = [n.severity for n in all_normalized]

    console.print("[cyan]Training Anomaly Detector (Isolation Forest)...[/]")
    anomaly = AnomalyDetector(contamination=0.15)
    anomaly.fit(feature_df)
    anomaly.save(str(model_path / "anomaly_detector.joblib"))
    console.print("[green]Anomaly detector saved.[/]")

    console.print("[cyan]Training Log Classifier (Random Forest)...[/]")
    classifier = LogClassifier()
    eval_results = classifier.train_with_eval(feature_df, labels)
    classifier.save(str(model_path / "classifier.joblib"))
    console.print(f"[green]Classifier accuracy: {eval_results['accuracy']:.4f}[/]")
    console.print(f"[green]Classifier F1: {eval_results['f1_weighted']:.4f}[/]")

    console.print("[cyan]Training Pattern Recognizer (TF-IDF + Cosine)...[/]")
    recognizer = PatternRecognizer()
    recognizer.fit(messages)
    recognizer.save(str(model_path / "pattern_recognizer.joblib"))
    console.print("[green]Pattern recognizer saved.[/]")

    console.print(f"\n[bold cyan]Training complete! Models saved to {model_dir}[/]")


@cli.command()
@click.argument("log_line", required=False)
@click.option("--file", "-f", "file_path", help="File with log lines to analyze")
@click.option("--format", "-fmt", "log_format", help="Log format hint")
@click.pass_context
def analyze(ctx: click.Context, log_line: str | None, file_path: str | None, log_format: str | None) -> None:
    """Analyze a log line or file for anomalies."""
    model_dir = ctx.obj["model_dir"]
    anomaly, classifier, _recognizer = _load_or_create_models(model_dir)

    parser = LogParser()
    normalizer = LogNormalizer()
    extractor = FeatureExtractor()

    lines: list[str] = []
    if file_path:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f if line.strip()]
    elif log_line:
        lines = [log_line]
    else:
        lines = [line.strip() for line in sys.stdin if line.strip()]

    if not lines:
        console.print("[yellow]No input provided.[/]")
        return

    for line in lines:
        parsed = parser.parse_line(line, log_format)
        if parsed is None:
            console.print(f"[red]Unparseable:[/] {line[:80]}")
            continue

        normalized = normalizer.normalize(parsed)
        features = extractor.extract(normalized)
        feat_df = pd.DataFrame([features])

        anomaly_score = anomaly.anomaly_score(feat_df)[0] if anomaly.is_fitted else 0.0
        is_anomaly = anomaly.predict(feat_df)[0] == 1 if anomaly.is_fitted else False
        severity_pred = classifier.predict(feat_df)[0] if classifier.is_fitted else "N/A"

        table = Table(box=box.ROUNDED, show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Severity", normalized.severity)
        table.add_row("Source", str(normalized.source))
        table.add_row("Type", normalized.log_type)
        table.add_row("Message", normalized.message[:100])
        table.add_row("Tags", ", ".join(normalized.tags) or "none")
        table.add_row("Anomaly Score", f"{anomaly_score:.4f}")
        table.add_row("Is Anomaly", str(is_anomaly))
        table.add_row("Predicted Severity", severity_pred)

        if classifier.is_fitted and is_anomaly:
            probs = classifier.predict_proba(feat_df)
            max_prob = max(probs[0]) if len(probs[0]) > 0 else 0
            table.add_row("Confidence", f"{max_prob:.4f}")

        console.print(table)
        console.print()


@cli.command()
@click.option("--file", "-f", "file_path", help="File to watch (stream)")
@click.pass_context
def watch(ctx: click.Context, file_path: str | None) -> None:
    """Watch a log file or stdin in real-time with dashboard."""
    model_dir = ctx.obj["model_dir"]
    anomaly, _, _ = _load_or_create_models(model_dir)

    source = file_path or "stdin"
    console.print(f"[cyan]Starting dashboard from {source}...[/]")

    dashboard = LogDashboard(
        anomaly_detector=anomaly if anomaly.is_fitted else None,
    )

    try:
        dashboard.run_live(source)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/]")


@cli.command()
@click.pass_context
def dashboard(ctx: click.Context) -> None:
    """Open an interactive text-based dashboard."""
    model_dir = ctx.obj["model_dir"]
    anomaly, _classifier, _recognizer = _load_or_create_models(model_dir)

    dashboard = LogDashboard(
        anomaly_detector=anomaly if anomaly.is_fitted else None,
    )

    console.print("[cyan]LogSentinel Dashboard[/]")
    console.print("[yellow]Enter log lines (Ctrl+C to quit):[/]")

    try:
        for line in sys.stdin:
            dashboard.ingest(line)
            table = Table(title="Live Statistics", box=box.SIMPLE)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="yellow")
            table.add_row("Total Logs", str(dashboard.total_count))
            table.add_row("Anomalies", str(dashboard.anomaly_count))
            table.add_row("Rate (logs/s)", f"{dashboard.total_count / max(time.time() - dashboard.start_time, 1):.2f}")
            console.clear()
            console.print(table)
            console.print("[bold red]Recent Anomalies:[/]" if dashboard.anomaly_history else "[green]No anomalies detected[/]")
            for a in list(dashboard.anomaly_history)[-5:]:
                console.print(f"  [red]{a['score']:.3f}[/] {a['severity']:7s} {a['message'][:70]}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard closed.[/]")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--model-dir", "-m", default=None, help="Directory with trained models")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, model_dir: str | None) -> None:
    """Start the FastAPI REST API server."""
    import uvicorn

    from src.api.server import create_app

    model_dir = model_dir or ctx.obj["model_dir"]
    app = create_app(model_dir=model_dir)
    console.print(f"[cyan]Starting LogSentinel API on {host}:{port}...[/]")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
