"""
Юнит-тесты детекции аномалий (app/anomaly.py), Total Load LSTM.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, TrainingData, AnomalyEvent, ForecastCache
from app.anomaly import detect_anomaly_for_fact


def _settings(**kwargs):
    """Настройки как в app.config (значения по умолчанию переопределяются в тестах)."""
    defaults = dict(
        anomaly_window_size=96,
        residual_sigma_k=3.0,
        mad_k=6.0,
        min_points_for_anomaly=20,
        anomaly_min_residual=0.15,
        anomaly_min_residual_pct=0.15,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _mk_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return Session()


def _seed_training(db, n=120, start=None, value=10.0):
    start = start or datetime(2020, 1, 1, 0, 0)
    for i in range(n):
        db.add(
            TrainingData(
                date=start + timedelta(minutes=15 * i),
                usage_kwh=float(value),
            )
        )
    db.commit()


def test_detect_anomaly_insufficient_history(monkeypatch):
    db = _mk_session()
    _seed_training(db, n=20)
    monkeypatch.setattr("app.anomaly.get_settings", lambda: _settings(
        anomaly_window_size=96,
        min_points_for_anomaly=48,
    ))
    event = detect_anomaly_for_fact(db, datetime(2020, 1, 1, 6, 0), 10.0)
    assert event.is_anomaly is False
    assert event.reason == "insufficient_history"


def test_detect_anomaly_normal(monkeypatch):
    db = _mk_session()
    start = datetime(2020, 1, 1, 0, 0)
    _seed_training(db, n=140, start=start, value=10.0)
    monkeypatch.setattr("app.anomaly.get_settings", lambda: _settings(
        anomaly_window_size=96,
        min_points_for_anomaly=20,
    ))
    monkeypatch.setattr("app.anomaly._predict_for_event_time", lambda *_: 10.1)
    for i in range(30):
        db.add(
            AnomalyEvent(
                event_time=start + timedelta(minutes=15 * i),
                actual_value=10.0,
                forecast_value=10.1,
                residual=0.1,
                residual_threshold=0.5,
                stat_score=0.1,
                rule_residual=False,
                rule_stat=False,
                is_anomaly=False,
                reason="normal",
            )
        )
    db.commit()
    event = detect_anomaly_for_fact(db, start + timedelta(minutes=15 * 140), 10.05)
    assert event.is_anomaly is False
    assert event.rule_residual is False
    assert event.reason == "realtime_forecast"


def test_detect_anomaly_outlier(monkeypatch):
    db = _mk_session()
    start = datetime(2020, 1, 1, 0, 0)
    _seed_training(db, n=160, start=start, value=10.0)
    monkeypatch.setattr("app.anomaly.get_settings", lambda: _settings(
        anomaly_window_size=96,
        min_points_for_anomaly=20,
        anomaly_min_residual=0.15,
        anomaly_min_residual_pct=0.15,
    ))
    monkeypatch.setattr("app.anomaly._predict_for_event_time", lambda *_: 10.0)
    for i in range(30):
        db.add(
            AnomalyEvent(
                event_time=start + timedelta(minutes=15 * i),
                actual_value=10.0,
                forecast_value=10.0,
                residual=0.2,
                residual_threshold=0.6,
                stat_score=0.2,
                rule_residual=False,
                rule_stat=False,
                is_anomaly=False,
                reason="normal",
            )
        )
    db.commit()
    event = detect_anomaly_for_fact(db, start + timedelta(minutes=15 * 160), 40.0)
    assert event.rule_residual is True
    assert event.is_anomaly is True
    assert event.reason is None


def test_committed_forecast_preferred_over_realtime(monkeypatch):
    """Если в forecast_cache есть committed-прогноз на fact_time — берётся он, не _predict."""
    db = _mk_session()
    start = datetime(2020, 1, 1, 0, 0)
    _seed_training(db, n=120, start=start, value=5.0)
    fact_time = start + timedelta(minutes=15 * 100)
    db.add(
        ForecastCache(
            computed_at=start,
            forecast_time=fact_time,
            value_kwh=7.5,
            is_committed=True,
        )
    )
    monkeypatch.setattr("app.anomaly.get_settings", lambda: _settings(
        anomaly_window_size=30,
        min_points_for_anomaly=10,
    ))
    monkeypatch.setattr("app.anomaly._predict_for_event_time", lambda *_: 999.0)

    for i in range(12):
        db.add(
            AnomalyEvent(
                event_time=start + timedelta(minutes=15 * i),
                actual_value=5.0,
                forecast_value=5.0,
                residual=0.1,
                residual_threshold=1.0,
                stat_score=0.1,
                rule_residual=False,
                rule_stat=False,
                is_anomaly=False,
                reason="normal",
            )
        )
    db.commit()

    event = detect_anomaly_for_fact(db, fact_time, 7.5)
    assert event.forecast_value == 7.5
    assert event.reason == "committed_forecast"
