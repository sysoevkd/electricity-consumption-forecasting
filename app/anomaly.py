"""
Онлайн-детекция аномалий фактических значений потребления (Total Load, 15 мин).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import TrainingData, AnomalyEvent, ForecastCache
from app.preprocessing import TARGET_COL, prepare_lstm_input, SEQ_LEN, N_FEATURES


EPS = 1e-9


@dataclass
class AnomalyCalcResult:
    event_time: datetime
    actual_value: float
    forecast_value: float
    residual: float
    residual_threshold: float
    stat_score: Optional[float]
    rule_residual: bool
    rule_stat: bool
    is_anomaly: bool
    reason: Optional[str]


def _robust_median_mad(values: np.ndarray) -> tuple[float, float]:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, mad


def _predict_for_event_time(db: Session, event_time: datetime) -> Optional[float]:
    """
    Строит LSTM-прогноз для event_time, используя только точки строго раньше event_time.
    """
    rows = (
        db.query(TrainingData.date, TrainingData.usage_kwh)
        .filter(TrainingData.date < event_time)
        .order_by(TrainingData.date.desc())
        .limit(SEQ_LEN)
        .all()
    )
    if len(rows) < SEQ_LEN:
        return None
    df = pd.DataFrame(
        [{"date": r.date, TARGET_COL: float(r.usage_kwh)} for r in reversed(rows)]
    )
    from app.main import model, scaler
    if model is None or scaler is None:
        return None
    from app.model import predict_multi
    X = prepare_lstm_input(df, scaler, seq_len=SEQ_LEN)
    # Используем первый шаг (+15 мин) как прогноз для детекции аномалий
    return predict_multi(model, X, scaler, n_features=N_FEATURES)[0]


def _lookup_committed_forecast(db: Session, fact_time: datetime) -> Optional[float]:
    """
    Ищет ранее сохранённый committed прогноз для fact_time в forecast_cache.
    Берёт самый ранний по computed_at — тот, что был сделан раньше всего до прихода факта.
    Возвращает None если committed прогноза нет.
    """
    row = (
        db.query(ForecastCache.value_kwh)
        .filter(
            ForecastCache.forecast_time == fact_time,
            ForecastCache.is_committed == True,
        )
        .order_by(ForecastCache.computed_at.asc())
        .first()
    )
    return float(row[0]) if row else None


def detect_anomaly_for_fact(db: Session, fact_time: datetime, fact_value: float) -> AnomalyEvent:
    """
    Детектирует аномалию для одного нового факта и сохраняет событие в anomaly_events.

    Источник прогноза (reason при нормальном исходе):
      - 'committed_forecast': прогноз был сохранён планировщиком ДО прихода факта —
        честный продовый цикл, модель не знала будущего.
      - 'realtime_forecast': committed прогноза не было (данные загружены задним числом),
        прогноз построен на месте по той же логике что и раньше.
    """
    existing = (
        db.query(AnomalyEvent)
        .filter(AnomalyEvent.event_time == fact_time)
        .first()
    )
    if existing:
        return existing

    settings = get_settings()

    # Приоритет 1: ранее сохранённый committed прогноз (настоящий прод-цикл)
    committed_value = _lookup_committed_forecast(db, fact_time)
    forecast_source = "committed_forecast" if committed_value is not None else "realtime_forecast"

    # Приоритет 2: fallback — строим прогноз на месте (для исторических данных)
    forecast_value = committed_value if committed_value is not None else _predict_for_event_time(db, fact_time)

    if forecast_value is None:
        event = AnomalyEvent(
            event_time=fact_time,
            actual_value=float(fact_value),
            forecast_value=float(fact_value),
            residual=0.0,
            residual_threshold=float("inf"),
            stat_score=None,
            rule_residual=False,
            rule_stat=False,
            is_anomaly=False,
            reason="insufficient_history",
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return event

    residual = abs(float(fact_value) - float(forecast_value))

    # 1) Dynamic residual threshold based on recent residual history
    # Включаем ВСЕ записи кроме insufficient_history (у них residual=0 без прогноза).
    # Важно: reason=NULL (аномалии) ДОЛЖНЫ участвовать в калибровке,
    # иначе порог строится только на «хороших» точках и занижается.
    residual_rows = (
        db.query(AnomalyEvent.residual)
        .filter(AnomalyEvent.event_time < fact_time)
        .filter(AnomalyEvent.reason.is_distinct_from("insufficient_history"))
        .order_by(AnomalyEvent.event_time.desc())
        .limit(settings.anomaly_window_size)
        .all()
    )
    residual_hist = np.array([float(r[0]) for r in residual_rows], dtype=float)
    if residual_hist.size >= settings.min_points_for_anomaly:
        res_med, res_mad = _robust_median_mad(residual_hist)
        adaptive_threshold = float(res_med + settings.residual_sigma_k * max(res_mad, EPS))
        # Итоговый порог = max(адаптивный, абсолютный floor, относительный floor).
        # Относительный floor: не флагировать если ошибка < X% факта (независимо от адаптивного).
        rel_floor = settings.anomaly_min_residual_pct * abs(float(fact_value))
        residual_threshold = max(adaptive_threshold, settings.anomaly_min_residual, rel_floor)
        rule_residual = residual > residual_threshold
    else:
        residual_threshold = float("inf")
        rule_residual = False

    # 2) Robust stat score for fact based on recent actual values
    actual_rows = (
        db.query(TrainingData.usage_kwh)
        .filter(TrainingData.date < fact_time)
        .order_by(TrainingData.date.desc())
        .limit(settings.anomaly_window_size)
        .all()
    )
    actual_hist = np.array([float(r[0]) for r in actual_rows], dtype=float)
    stat_score: Optional[float] = None
    if actual_hist.size >= settings.min_points_for_anomaly:
        med, mad = _robust_median_mad(actual_hist)
        robust_sigma = 1.4826 * max(mad, EPS)
        stat_score = abs((float(fact_value) - med) / robust_sigma)
        rule_stat = stat_score > settings.mad_k
    else:
        rule_stat = False

    # Аномалия = модель существенно ошиблась (rule_residual).
    # rule_stat служит дополнительным информационным сигналом, но не является
    # обязательным условием: 24-часовое окно не учитывает недельную сезонность,
    # поэтому итоговое решение основано исключительно на rule_residual.
    is_anomaly = bool(rule_residual)
    # reason кодирует и результат, и источник прогноза:
    #   None                  — аномалия (любой источник)
    #   "committed_forecast"  — норма, прогноз был сохранён заранее (прод-цикл)
    #   "realtime_forecast"   — норма, прогноз построен на месте (исторические данные)
    if rule_residual:
        reason = None
    else:
        reason = forecast_source  # "committed_forecast" или "realtime_forecast"

    event = AnomalyEvent(
        event_time=fact_time,
        actual_value=float(fact_value),
        forecast_value=float(forecast_value),
        residual=float(residual),
        residual_threshold=float(residual_threshold),
        stat_score=None if stat_score is None else float(stat_score),
        rule_residual=bool(rule_residual),
        rule_stat=bool(rule_stat),
        is_anomaly=is_anomaly,
        reason=reason,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event

