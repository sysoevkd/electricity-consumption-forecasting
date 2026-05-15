"""
Тесты Pydantic-схем API Total Load LSTM (app/schemas.py).
"""
import pytest
from pydantic import ValidationError

from app.schemas import (
    DataPoint,
    PredictionRequest,
    PredictionResponse,
    DataUploadResponse,
    TrainingDataStatsResponse,
    HealthResponse,
    MetricsResponse,
    AnomalyStatsResponse,
)


def _point(ts: str = "2024-01-01T00:00:00", kwh: float = 2.5) -> dict:
    return {"date": ts, "consumption_kWh": kwh}


class TestDataPoint:
    def test_valid(self):
        p = DataPoint(date="2024-06-15T12:00:00", consumption_kWh=1.23)
        assert p.consumption_kWh == 1.23

    def test_consumption_required(self):
        with pytest.raises(ValidationError):
            DataPoint(date="2024-01-01T00:00:00")


class TestPredictionRequest:
    def test_min_96_points_ok(self):
        payload = {"data": [_point() for _ in range(96)]}
        req = PredictionRequest(**payload)
        assert len(req.data) == 96

    def test_95_points_rejected(self):
        payload = {"data": [_point() for _ in range(95)]}
        with pytest.raises(ValidationError):
            PredictionRequest(**payload)


class TestPredictionResponse:
    def test_four_horizons(self):
        r = PredictionResponse(
            forecast={"+15min": 1.0, "+30min": 1.1, "+45min": 1.2, "+60min": 1.3},
            forecast_times=[
                "2024-01-01T00:15:00",
                "2024-01-01T00:30:00",
                "2024-01-01T00:45:00",
                "2024-01-01T01:00:00",
            ],
        )
        assert len(r.forecast) == 4
        assert len(r.forecast_times) == 4


class TestHealthResponse:
    def test_valid(self):
        r = HealthResponse(
            status="работает",
            model_loaded=True,
            database_connected=True,
        )
        assert r.model_loaded is True
        assert r.timestamp is not None


class TestMetricsResponse:
    def test_optional_fields(self):
        r = MetricsResponse()
        assert r.mae is None

    def test_with_values(self):
        r = MetricsResponse(mae=0.17, rmse=0.32, r2=0.97, training_samples=2000)
        assert r.mae == 0.17


class TestTrainingDataStatsResponse:
    def test_valid(self):
        r = TrainingDataStatsResponse(
            total_rows=100,
            first_date="2024-01-01T00:00:00",
            last_date="2024-01-02T00:00:00",
            enough_for_predict=True,
            enough_for_backtest=False,
        )
        assert r.enough_for_predict is True


class TestAnomalyStatsResponse:
    def test_valid(self):
        r = AnomalyStatsResponse(
            total_events=100,
            anomaly_events=5,
            anomaly_rate=0.05,
        )
        assert r.anomaly_rate == 0.05


class TestDataUploadResponse:
    def test_valid(self):
        r = DataUploadResponse(message="ok", rows_uploaded=10, total_rows=200)
        assert r.total_rows == 200
