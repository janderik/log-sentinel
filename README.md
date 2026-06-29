<div align="center">
  <h1>🔍 LogSentinel</h1>
  <p><strong>ML-Powered Log Analyzer — Anomaly Detection, Classification & Pattern Recognition</strong></p>

  ![Python](https://img.shields.io/badge/python-3.10%2B-blue)
  ![CI](https://github.com/logsentinel/log-sentinel/actions/workflows/ci.yml/badge.svg)
  ![License](https://img.shields.io/badge/license-MIT-green)
  [![codecov](https://codecov.io/gh/logsentinel/log-sentinel/branch/main/graph/badge.svg)](https://codecov.io/gh/logsentinel/log-sentinel)
</div>

---

## Features

- **Multi-format Log Parsing** — Apache/NGINX combined, syslog (RFC 3164), Windows Event (EVTX-style), JSON, custom regex
- **Unsupervised Anomaly Detection** — Isolation Forest identifies outliers in log patterns without labeled data
- **Supervised Log Classification** — Random Forest categorizes logs by severity and type
- **Pattern Recognition** — TF-IDF vectorization + cosine similarity to find similar log messages
- **Real-time API** — FastAPI endpoint for log submission and instant analysis
- **Interactive Dashboard** — Rich-powered terminal dashboard with live anomaly tracking
- **CLI** — `train`, `analyze`, `watch`, `dashboard` subcommands
- **Docker Support** — Containerized deployment with docker-compose
- **Model Persistence** — All models serialized with joblib for production reuse

## Architecture

```
┌────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│  Log File  │───▶│  LogParser   │───▶│  LogNormalizer │───▶│ FeatureExtr. │
│  /stdin    │    │  (regex)     │    │  (unified      │    │  (temporal,  │
│  /API      │    │              │    │   schema)      │    │   text, cat) │
└────────────┘    └──────────────┘    └────────────────┘    └──────┬───────┘
                                                                   │
         ┌─────────────────────────────────────────────────────────┘
         ▼                    ▼                    ▼
┌────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│AnomalyDetector │  │  LogClassifier   │  │  PatternRecognizer   │
│(IsolationForest)│  │(RandomForest)    │  │(TF-IDF + Cosine Sim)│
│ → anomaly score │  │ → severity pred  │  │ → similar patterns  │
└────────────────┘  └──────────────────┘  └──────────────────────┘
```

## Installation

```bash
# Clone & install
git clone https://github.com/logsentinel/log-sentinel.git
cd log-sentinel
pip install -r requirements.txt
pip install -e .
```

## Usage

### CLI

```bash
# Train models on sample data
logsentinel train --input-dir ./examples/logs --model-dir ./models

# Analyze a single log line
logsentinel analyze '192.168.1.1 - frank [10/Oct/2023:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326'

# Analyze from file
logsentinel analyze --file ./examples/logs/sample_apache.log

# Watch a log file in real-time dashboard
logsentinel watch --file ./examples/logs/sample_apache.log

# Interactive dashboard (type or pipe logs)
logsentinel dashboard
```

### REST API

```bash
# Start the API server
uvicorn src.api.server:create_app --factory --port 8000

# Analyze a log line
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"line": "192.168.1.1 - frank [10/Oct/2023:13:55:36 -0700] \"GET /index.html HTTP/1.1\" 200 2326"}'

# Batch analysis
curl -X POST http://localhost:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"lines": ["line1", "line2"], "format": "apache_combined"}'
```

### Docker

```bash
docker-compose up --build
```

## Machine Learning Models

### Anomaly Detector (Isolation Forest)
- **Algorithm**: `sklearn.ensemble.IsolationForest`
- **Approach**: Unsupervised — isolates anomalies by randomly partitioning the feature space
- **Features**: 25+ numerical features (temporal, text entropy, ratios, categorical encodings)
- **Output**: Normalized anomaly score (0–1), binary anomaly flag
- **Contamination**: Default 10%, configurable

### Log Classifier (Random Forest)
- **Algorithm**: `sklearn.ensemble.RandomForestClassifier`
- **Approach**: Supervised — trained on log severity labels
- **Classes**: DEBUG, INFO, WARN, ERROR, CRITICAL
- **Metrics**: Accuracy, weighted F1, per-class precision/recall/f1
- **Class balancing**: `class_weight='balanced'`

### Pattern Recognizer (TF-IDF + Cosine Similarity)
- **Vectorizer**: `sklearn.feature_extraction.text.TfidfVectorizer` (character n-grams 2–4)
- **Similarity**: Cosine similarity between TF-IDF vectors
- **Output**: Top-k most similar historical log messages with similarity scores
- **Clustering**: Threshold-based grouping of similar patterns

## Project Structure

```
log-sentinel/
├── src/
│   ├── parser/          # Log parsing & normalization
│   ├── features/        # Feature extraction & vectorization
│   ├── models/          # ML models (anomaly, classifier, pattern)
│   ├── api/             # FastAPI REST server & routes
│   └── visualization/   # Terminal dashboard
├── tests/               # pytest suite
├── examples/logs/       # Sample logs (Apache, syslog, Windows)
└── .github/workflows/   # CI pipeline
```

## License

MIT
