"""
Интеграционные тесты API Total Load LSTM.

Требования: запущенный сервис (docker compose up -d postgres api) и модель в models/.
Переменная API_BASE_URL (по умолчанию http://localhost:8000).

Запуск только интеграции:
  pytest tests/test_api_integration.py -v

Полный набор без Docker (юнит-тесты):
  pytest tests/ -v --ignore=tests/test_api_integration.py
"""
import io
import time

import httpx
import pytest

from tests.conftest import BASE_URL

HTTP_TIMEOUT = 30.0
HTTP_TIMEOUT_LONG = 120.0
HEALTH_WAIT_TIMEOUT = 15.0


def _url(path: str) -> str:
    return f"{BASE_URL.rstrip('/')}{path}"


def _wait_health(api_base_url: str, timeout: float = HEALTH_WAIT_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{api_base_url}/health", timeout=3.0)
            if r.status_code == 200 and r.json().get("model_loaded"):
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


@pytest.fixture(scope="module")
def ensure_api_ready():
    if not _wait_health(BASE_URL.rstrip("/")):
        pytest.skip(
            f"API недоступен за {HEALTH_WAIT_TIMEOUT} с: {BASE_URL}. "
            "Запустите: docker compose up -d postgres api"
        )


def test_health(ensure_api_ready):
    r = httpx.get(_url("/health"), timeout=HTTP_TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert data.get("database_connected") is True
    assert data.get("model_loaded") is True


def test_root_total_load(ensure_api_ready):
    r = httpx.get(_url("/"), timeout=HTTP_TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert "LSTM" in data.get("сервис", "")
    assert data.get("документация") == "/docs"


def test_model_metrics(ensure_api_ready):
    r = httpx.get(_url("/model/metrics"), timeout=HTTP_TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert "mae" in data


def test_predict_multi_step(ensure_api_ready, lstm_predict_payload):
    r = httpx.post(_url("/predict"), json=lstm_predict_payload, timeout=HTTP_TIMEOUT)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "forecast" in data
    assert len(data["forecast"]) == 4
    assert "+15min" in data["forecast"]
    assert "forecast_times" in data
    assert len(data["forecast_times"]) == 4


def test_predict_too_few_points(ensure_api_ready):
    payload = {"data": [{"date": "2024-01-01T00:15:00", "consumption_kWh": 1.0}] * 50}
    r = httpx.post(_url("/predict"), json=payload, timeout=HTTP_TIMEOUT)
    assert r.status_code == 422


def test_data_stats(ensure_api_ready):
    r = httpx.get(_url("/data/stats"), timeout=HTTP_TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert "total_rows" in data
    assert "enough_for_predict" in data


def test_forecast_compute_returns_four_saved(ensure_api_ready):
    r = httpx.post(_url("/forecast/compute"), timeout=HTTP_TIMEOUT_LONG)
    if r.status_code == 400:
        pytest.skip("Недостаточно строк в training_data для прогноза: " + r.text)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("saved") == 4
    assert "+15min" in body.get("values", {})


def test_forecast_cache_structure(ensure_api_ready):
    r = httpx.get(_url("/forecast/cache"), timeout=HTTP_TIMEOUT)
    assert r.status_code == 200
    cache = r.json()
    assert "data" in cache
    assert "count" in cache
    if cache["count"]:
        row = cache["data"][0]
        assert "forecast_time" in row
        assert "value_kwh" in row


def test_prometheus_metrics(ensure_api_ready):
    r = httpx.get(_url("/metrics"), timeout=HTTP_TIMEOUT)
    assert r.status_code == 200
    assert "model_" in r.text or "http_" in r.text


def test_anomalies_stats(ensure_api_ready):
    r = httpx.get(_url("/anomalies/stats"), timeout=HTTP_TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert "total_events" in data
    assert "anomaly_rate" in data


def test_anomalies_list(ensure_api_ready):
    r = httpx.get(_url("/anomalies?limit=10"), timeout=HTTP_TIMEOUT)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_upload_csv(ensure_api_ready, small_upload_csv):
    r = httpx.post(
        _url("/upload-data"),
        files={"file": ("chunk.csv", io.BytesIO(small_upload_csv.encode("utf-8")), "text/csv")},
        timeout=HTTP_TIMEOUT_LONG,
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "rows_uploaded" in data
    assert "total_rows" in data
