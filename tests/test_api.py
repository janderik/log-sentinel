from fastapi.testclient import TestClient

from src.api.server import create_app

APACHE_500 = '203.0.113.5 - - [10/Oct/2023:14:40:05 -0700] "POST /api/login HTTP/1.1" 500 255'


def _client(tmp_path) -> TestClient:
    # Empty model dir: parsing and features must still work without ML models.
    return TestClient(create_app(model_dir=str(tmp_path / "no-models")))


def test_analyze_without_trained_models(tmp_path) -> None:
    with _client(tmp_path) as client:
        response = client.post("/analyze", json={"line": APACHE_500})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["parsed"]["severity"] == "ERROR"
        assert "server_error" in body["normalized"]["tags"]
        assert body["anomaly_score"] is None


def test_analyze_batch_and_status(tmp_path) -> None:
    with _client(tmp_path) as client:
        batch = client.post("/analyze/batch", json={"lines": [APACHE_500, "not a log line at all"]})
        assert batch.status_code == 200, batch.text
        assert batch.json()["total_count"] == 2
        assert client.get("/models/status").json() == {
            "anomaly_detector": False,
            "classifier": False,
            "pattern_recognizer": False,
        }
