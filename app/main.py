"""
FastAPI сервис прогнозирования потребления электроэнергии (Total Load, 15 мин).
Модель: LSTMModelOptuna, Direct Multi-Step — SEQ_LEN=96 × N_FEATURES=6 → 4 значения (+15/+30/+45/+60 мин).
"""
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional, Any, List

from io import StringIO

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Query
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.orm import Session
from sqlalchemy import func, text

from app.config import get_settings
from app.model import load_model, load_scaler, predict_multi
from app.preprocessing import (
    prepare_lstm_input, TARGET_COL, SEQ_LEN, HORIZON, N_FEATURES
)
from app.database import (
    get_db, init_db,
    TrainingData,
    ForecastCache, AnomalyEvent, BacktestCache,
    SessionLocal,
)
from app.anomaly import detect_anomaly_for_fact
from app.schemas import (
    PredictionRequest, PredictionResponse,
    DataUploadResponse, TrainingDataStatsResponse,
    AnomalyEventResponse, AnomalyStatsResponse,
    HealthResponse, MetricsResponse,
)
from app.metrics import (
    record_prediction, record_data_upload,
    record_anomaly_event,
    record_model_load, update_training_data_count, PREDICTION_LATENCY,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

model:  Optional[Any] = None
scaler: Optional[Any] = None

# Минимум точек для прогноза — SEQ_LEN (96 = 24 ч)
MIN_POINTS_FOR_PREDICT = SEQ_LEN  # 96

# Метки горизонтов
STEP_LABELS = ["+15min", "+30min", "+45min", "+60min"]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _parse_upload_date(series: pd.Series) -> pd.Series:
    """Парсинг даты из CSV: ISO и %d/%m/%Y %H:%M."""
    def parse_one(val):
        if pd.isna(val):
            return pd.NaT
        s = str(val).strip()
        try:
            return pd.to_datetime(s, format="%d/%m/%Y %H:%M")
        except Exception:
            return pd.to_datetime(s)
    return series.apply(parse_one)


def _resolve_path(path_str: str) -> Path:
    """Разрешает относительный путь от корня проекта."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent.parent / p


def load_model_artifacts():
    """Загружает LSTMModelOptuna и scaler."""
    global model, scaler

    model_path  = _resolve_path(settings.model_path)
    params_path = _resolve_path(settings.params_path)
    scaler_path = _resolve_path(settings.scaler_path)

    if not model_path.exists():
        logger.warning(f"Модель не найдена: {model_path}. Скопируйте файлы и вызовите POST /model/reload.")
        return
    if not scaler_path.exists():
        logger.warning(f"Scaler не найден: {scaler_path}.")
        return

    try:
        model  = load_model(str(model_path), str(params_path) if params_path.exists() else None)
        scaler = load_scaler(str(scaler_path))
        record_model_load()
        logger.info("LSTMModelOptuna и scaler успешно загружены")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        model = scaler = None


def _predict(work_df: pd.DataFrame) -> List[float]:
    """
    Один шаг инференса: последние SEQ_LEN строк -> список из 4 прогнозов [+15, +30, +45, +60 мин].
    """
    X = prepare_lstm_input(work_df, scaler, seq_len=MIN_POINTS_FOR_PREDICT)
    return predict_multi(model, X, scaler, n_features=N_FEATURES)


# ---------------------------------------------------------------------------
# Кэш прогноза и backtest
# ---------------------------------------------------------------------------

FORECAST_HISTORY_DAYS = 30  # хранить историю прогнозов за последние N дней
BACKTEST_AUTO_STEPS = 300   # сколько точек покрывать бэктестом при авто-запуске после upload-data


def compute_and_save_forecast(db: Session) -> dict:
    """
    Прогноз на следующий час: один вызов модели → 4 значения (+15/+30/+45/+60 мин).
    Каждый запуск добавляет 4 строки в forecast_cache (не перезаписывает историю).
    Это позволяет при загрузке нового факта найти ранее сохранённый committed прогноз
    и сравнить факт с ним.
    Старые записи (> FORECAST_HISTORY_DAYS) удаляются для контроля размера таблицы.
    """
    if model is None or scaler is None:
        raise ValueError("Модель или scaler не загружены")

    rows = (
        db.query(TrainingData.date, TrainingData.usage_kwh)
        .order_by(TrainingData.date.desc())
        .limit(MIN_POINTS_FOR_PREDICT)
        .all()
    )
    if len(rows) < MIN_POINTS_FOR_PREDICT:
        raise ValueError(
            f"В БД только {len(rows)} строк. Нужно минимум {MIN_POINTS_FOR_PREDICT}."
        )

    df = pd.DataFrame(
        [{"date": r.date, TARGET_COL: float(r.usage_kwh)} for r in reversed(rows)]
    )
    computed_at = datetime.utcnow()
    last_date   = pd.Timestamp(df["date"].iloc[-1]).to_pydatetime()

    preds = _predict(df)  # [+15min, +30min, +45min, +60min]

    # Чистка устаревших прогнозов (старше FORECAST_HISTORY_DAYS дней)
    cutoff = computed_at - timedelta(days=FORECAST_HISTORY_DAYS)
    db.query(ForecastCache).filter(ForecastCache.computed_at < cutoff).delete()

    # Добавляем новые прогнозы, не удаляя предыдущие — они нужны для committed-сравнения
    for i, pred_value in enumerate(preds):
        forecast_time = last_date + timedelta(minutes=15 * (i + 1))
        db.add(ForecastCache(
            computed_at=computed_at,
            forecast_time=forecast_time,
            value_kwh=pred_value,
            is_committed=True,
        ))
    db.commit()
    # Обновляем Prometheus: каждый запуск планировщика = 4 успешных прогноза
    for pred_value in preds:
        record_prediction(latency=0.0, value=pred_value, success=True)
    logger.info(f"Прогноз сохранён: 4 шага (+15/+30/+45/+60 мин), computed_at={computed_at.isoformat()}")
    return {
        "saved":               4,
        "step_minutes":        15,
        "computed_at":         computed_at.isoformat(),
        "last_fact_date":      last_date.isoformat(),
        "forecast_time_first": (last_date + timedelta(minutes=15)).isoformat(),
        "forecast_time_last":  (last_date + timedelta(minutes=60)).isoformat(),
        "values":              {STEP_LABELS[i]: round(v, 4) for i, v in enumerate(preds)},
    }


def compute_and_save_backtest(
    db: Session,
    steps: int = 25,
    from_time: Optional[datetime] = None,
) -> dict:
    """
    Backtest: для каждой из `steps` точек прогнозирует все 4 шага
    (+15, +30, +45, +60 мин) и сравнивает с фактом.
    from_time — начало тестового окна; если None — берутся последние steps точек.
    Аномалии детектируются только по шагу 1 (+15 мин).
    """
    if model is None or scaler is None:
        raise ValueError("Модель или scaler не загружены")

    if from_time is not None:
        # Загружаем: SEQ_LEN точек до from_time + steps тестовых + 4 для горизонта
        rows = (
            db.query(TrainingData.date, TrainingData.usage_kwh)
            .filter(TrainingData.date >= from_time - timedelta(minutes=15 * MIN_POINTS_FOR_PREDICT))
            .order_by(TrainingData.date.asc())
            .limit(MIN_POINTS_FOR_PREDICT + steps + 4)
            .all()
        )
    else:
        need_rows = MIN_POINTS_FOR_PREDICT + steps + 4
        rows = (
            db.query(TrainingData.date, TrainingData.usage_kwh)
            .order_by(TrainingData.date.desc())
            .limit(need_rows)
            .all()
        )
        rows = list(reversed(rows))

    if len(rows) < MIN_POINTS_FOR_PREDICT + steps + 1:
        raise ValueError(
            f"В БД только {len(rows)} строк. Нужно минимум {MIN_POINTS_FOR_PREDICT + steps + 1} для backtest."
        )

    df = pd.DataFrame(
        [{"date": r.date, TARGET_COL: float(r.usage_kwh)} for r in rows]
    ).reset_index(drop=True)

    computed_at = datetime.utcnow()
    test_start  = len(df) - steps - 1
    test_end    = len(df) - 1   # не включается

    # step1_rows используется для аномалий и диапазона удаления
    step1_rows: list[tuple] = []
    # all_step_rows: (step_num, point_time, forecast, actual, err, abs_err)
    all_step_rows: list[tuple] = []

    HORIZON = 4
    for i in range(test_start, test_end):
        hist  = df.iloc[i - MIN_POINTS_FOR_PREDICT : i].copy()
        preds = _predict(hist)

        for k in range(HORIZON):
            actual_idx = i + k
            if actual_idx >= len(df):
                break
            forecast_val = float(preds[k])
            actual_val   = float(df.iloc[actual_idx][TARGET_COL])
            point_time   = pd.Timestamp(df.iloc[actual_idx]["date"]).to_pydatetime()
            err          = actual_val - forecast_val
            abs_err      = abs(err)
            all_step_rows.append((k + 1, point_time, forecast_val, actual_val, err, abs_err))
            if k == 0:
                step1_rows.append((point_time, forecast_val, actual_val, err, abs_err))

    # Удаляем только строки в диапазоне текущего вычисления — старые данные остаются.
    if all_step_rows:
        bt_range_start = min(r[1] for r in all_step_rows)
        bt_range_end   = max(r[1] for r in all_step_rows)
        db.query(BacktestCache).filter(
            BacktestCache.point_time >= bt_range_start,
            BacktestCache.point_time <= bt_range_end,
        ).delete()

    # Удаляем аномалии в диапазоне backtest чтобы не дублировать с online-событиями
    if step1_rows:
        range_start = step1_rows[0][0]
        range_end   = step1_rows[-1][0]
        db.query(AnomalyEvent).filter(
            AnomalyEvent.event_time >= range_start,
            AnomalyEvent.event_time <= range_end,
        ).delete()

    # Сохраняем все шаги в backtest_cache
    for step_num, point_time, forecast_val, actual_val, err, abs_err in all_step_rows:
        db.add(BacktestCache(
            computed_at=computed_at,
            point_time=point_time,
            step_num=step_num,
            forecast_value=forecast_val,
            actual_value=actual_val,
            error=err,
            abs_error=abs_err,
        ))

    # Аномалии — только по шагу 1 (+15 мин)
    residual_hist: list[float] = []
    for point_time, forecast_val, actual_val, err, abs_err in step1_rows:
        if len(residual_hist) >= 1:
            r_arr    = np.array(residual_hist[-settings.anomaly_window_size:], dtype=float)
            r_med    = float(np.median(r_arr))
            r_mad    = float(np.median(np.abs(r_arr - r_med)))
            adaptive = float(r_med + settings.residual_sigma_k * max(r_mad, 1e-9))
            rel_floor = settings.anomaly_min_residual_pct * abs(float(actual_val))
            res_th   = max(adaptive, settings.anomaly_min_residual, rel_floor)
            rule_r   = float(abs_err) > res_th
        else:
            rel_floor = settings.anomaly_min_residual_pct * abs(float(actual_val))
            res_th = float(max(abs_err * 1.25, settings.anomaly_min_residual, rel_floor))
            rule_r = False

        hist_actual = (
            df.loc[df["date"] < point_time, TARGET_COL]
            .tail(settings.anomaly_window_size).astype(float).to_numpy()
        )
        if hist_actual.size >= settings.min_points_for_anomaly:
            a_med   = float(np.median(hist_actual))
            a_mad   = float(np.median(np.abs(hist_actual - a_med)))
            robust_sigma = 1.4826 * max(a_mad, 1e-9)
            stat_sc = abs((float(actual_val) - a_med) / robust_sigma)
            rule_s  = stat_sc > settings.mad_k
        else:
            stat_sc = None
            rule_s  = False

        is_anomaly = bool(rule_r)
        db.add(AnomalyEvent(
            event_time=point_time,
            actual_value=float(actual_val),
            forecast_value=float(forecast_val),
            residual=float(abs_err),
            residual_threshold=float(res_th),
            stat_score=None if stat_sc is None else float(stat_sc),
            rule_residual=bool(rule_r),
            rule_stat=bool(rule_s),
            is_anomaly=is_anomaly,
            reason="backtest",
        ))
        record_anomaly_event(is_anomaly=is_anomaly, residual=float(abs_err), stat_score=stat_sc)
        residual_hist.append(float(abs_err))

    db.commit()

    # Обновляем per-step Prometheus метрики
    from collections import defaultdict
    from app.metrics import update_step_metrics
    step_errors: dict = defaultdict(list)
    for step_num, _, _, actual_val, err, abs_err in all_step_rows:
        pct = abs(err / actual_val * 100) if actual_val != 0 else 0.0
        step_errors[step_num].append((abs_err, err ** 2, pct))
    update_step_metrics(step_errors)

    return {
        "saved":       len(step1_rows),
        "steps_total": len(all_step_rows),
        "horizon":     HORIZON,
        "step":        "+15/+30/+45/+60 мин",
        "computed_at": computed_at.isoformat(),
        "from_time":   step1_rows[0][0].isoformat() if step1_rows else None,
        "to_time":     step1_rows[-1][0].isoformat() if step1_rows else None,
    }


def _run_forecast_job():
    """Фоновая задача: обновление forecast_cache каждые 15 мин."""
    db = SessionLocal()
    try:
        compute_and_save_forecast(db)
    except Exception as e:
        logger.warning("Не удалось обновить прогноз: %s", e)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Lifespan & app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск ML сервиса (Total Load, LSTMModelOptuna, Direct Multi-Step, 15 мин)...")
    init_db()
    load_model_artifacts()

    # Инициализируем счётчик записей в БД при старте
    _db = SessionLocal()
    try:
        count = _db.query(func.count(TrainingData.id)).scalar() or 0
        update_training_data_count(count)
    finally:
        _db.close()

    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(_run_forecast_job, "interval", minutes=15, id="forecast_15m")
    scheduler.start()
    scheduler.add_job(
        _run_forecast_job, "date",
        run_date=datetime.utcnow() + timedelta(minutes=1),
        id="forecast_first",
    )
    logger.info("Планировщик прогноза запущен (каждые 15 мин + первый через 1 мин)")
    yield
    scheduler.shutdown(wait=False)
    logger.info("Остановка ML сервиса...")


app = FastAPI(
    title="Сервис прогнозирования потребления (Total Load, LSTM)",
    description=(
        "LSTMModelOptuna, Direct Multi-Step: вход — последние 96 точек (24 ч, 6 признаков), "
        "выход — 4 значения (+15/+30/+45/+60 мин). "
        "Модель обновляется вручную: скопируйте .pt/.joblib в models/ → POST /model/reload."
    ),
    version="5.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, summary="Проверка состояния", tags=["Система"])
async def health_check(db: Session = Depends(get_db)):
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    ready = model is not None and scaler is not None and db_ok
    return HealthResponse(
        status="работает" if ready else "ограничен",
        model_loaded=model is not None and scaler is not None,
        database_connected=db_ok,
    )


@app.get("/metrics", summary="Метрики Prometheus", tags=["Система"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/model/reload", summary="Перезагрузить модель с диска", tags=["Модель"])
async def reload_model():
    """
    Перезагружает LSTMModelOptuna и scaler из папки models/.
    Вызывайте после обновления файлов .pt / .joblib.
    """
    load_model_artifacts()
    return {
        "status":       "ok",
        "model_loaded": model is not None and scaler is not None,
        "model_path":   settings.model_path,
        "scaler_path":  settings.scaler_path,
    }


def _metrics_from_backtest(db: Session) -> MetricsResponse:
    """
    MAE, RMSE, MSE, R² по данным backtest_cache (шаг 1 = +15 мин: прогноз vs факт).
    Заполняется после POST /backtest/compute. Если backtest пуст — все метрики None.
    training_samples — число строк в training_data.
    """
    total_rows = db.query(func.count(TrainingData.id)).scalar() or 0

    row = (
        db.query(
            func.count(BacktestCache.id),
            func.avg(BacktestCache.abs_error),
            func.avg(BacktestCache.error * BacktestCache.error),
            func.sum(BacktestCache.error * BacktestCache.error),
            func.avg(BacktestCache.actual_value),
            func.sum(BacktestCache.actual_value * BacktestCache.actual_value),
            func.max(BacktestCache.computed_at),
        )
        .filter(BacktestCache.step_num == 1)
        .one()
    )

    n = int(row[0] or 0)
    if n == 0:
        return MetricsResponse(
            mae=None,
            rmse=None,
            r2=None,
            mse=None,
            last_updated=None,
            training_samples=total_rows,
        )

    mae = float(row[1])
    mse = float(row[2])
    rmse = mse ** 0.5
    ss_res = float(row[3])
    mean_y = float(row[4])
    sum_y2 = float(row[5] or 0.0)
    last_up = row[6]

    ss_tot = sum_y2 - n * mean_y * mean_y
    if ss_tot > 1e-12:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = None

    return MetricsResponse(
        mae=round(mae, 6),
        rmse=round(rmse, 6),
        mse=round(mse, 6),
        r2=round(r2, 6) if r2 is not None else None,
        last_updated=last_up,
        training_samples=total_rows,
    )


@app.get("/model/metrics", response_model=MetricsResponse, summary="Метрики модели", tags=["Модель"])
async def get_model_metrics(db: Session = Depends(get_db)):
    """
    Качество модели по последнему backtest: сравнение прогноза и факта (шаг +15 мин).
    Данные из `backtest_cache` после вызова `POST /backtest/compute`.
    Если backtest не запускался — поля mae/rmse/mse/r2 будут null.
    """
    return _metrics_from_backtest(db)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Прогноз на 1 час вперёд (4 × 15 мин)",
    tags=["Прогнозирование"],
)
async def predict(request: PredictionRequest):
    """
    LSTMModelOptuna Direct Multi-Step: принимает минимум 96 точек (date, consumption_kWh),
    возвращает 4 прогнозных значения (+15/+30/+45/+60 мин от последней точки).
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Модель или scaler не загружены. Скопируйте файлы в models/ и вызовите POST /model/reload.",
        )

    start_time = time.time()
    try:
        rows = [{"date": dp.date, TARGET_COL: dp.consumption_kWh} for dp in request.data]
        df   = pd.DataFrame(rows)

        preds = _predict(df)

        # Вычисляем метки времени от последней известной точки
        last_dt_raw = request.data[-1].date
        if last_dt_raw:
            try:
                last_dt = datetime.fromisoformat(str(last_dt_raw))
            except Exception:
                last_dt = datetime.utcnow()
        else:
            last_dt = datetime.utcnow()

        forecast_times = [
            (last_dt + timedelta(minutes=15 * (i + 1))).isoformat()
            for i in range(HORIZON)
        ]

        latency = time.time() - start_time
        record_prediction(latency, preds[0], success=True)

        return PredictionResponse(
            forecast={STEP_LABELS[i]: round(v, 4) for i, v in enumerate(preds)},
            forecast_times=forecast_times,
        )
    except ValueError as e:
        record_prediction(time.time() - start_time, 0, success=False)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        record_prediction(time.time() - start_time, 0, success=False)
        logger.exception("Ошибка прогнозирования")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/cache", summary="Кэш прогноза (4 шага на 1 ч)", tags=["Прогнозирование"])
async def forecast_cache_get(db: Session = Depends(get_db)):
    """Последний сохранённый прогноз из forecast_cache (+15/+30/+45/+60 мин)."""
    rows = (
        db.query(ForecastCache.forecast_time, ForecastCache.value_kwh, ForecastCache.computed_at)
        .order_by(ForecastCache.forecast_time)
        .all()
    )
    return {
        "count":       len(rows),
        "step_minutes": 15,
        "data": [
            {
                "forecast_time": r.forecast_time.isoformat(),
                "value_kwh":     round(float(r.value_kwh), 4),
                "computed_at":   r.computed_at.isoformat(),
            }
            for r in rows
        ],
    }


@app.post("/forecast/compute", summary="Пересчитать прогноз из БД", tags=["Прогнозирование"])
async def forecast_compute(db: Session = Depends(get_db)):
    """Вычисляет прогноз на +15/+30/+45/+60 мин по последним 96 точкам из training_data."""
    try:
        return compute_and_save_forecast(db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Ошибка расчёта прогноза")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest/compute", summary="Backtest прогноз vs факт (+15 мин)", tags=["Прогнозирование"])
async def backtest_compute(
    steps:     int               = Query(25, ge=5, le=2000, description="Число тестовых точек"),
    from_time: Optional[datetime] = Query(None, description="Начало тестового окна (если не указано — последние steps точек)"),
    db:        Session            = Depends(get_db),
):
    try:
        return compute_and_save_backtest(db, steps=steps, from_time=from_time)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Ошибка backtest")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest/cache", summary="Кэш backtest", tags=["Прогнозирование"])
async def backtest_cache_get(db: Session = Depends(get_db)):
    rows = (
        db.query(
            BacktestCache.point_time, BacktestCache.forecast_value,
            BacktestCache.actual_value, BacktestCache.error, BacktestCache.abs_error,
        ).order_by(BacktestCache.point_time).all()
    )
    return {
        "count": len(rows),
        "data": [
            {
                "time":         r.point_time.isoformat(),
                "forecast_kwh": round(float(r.forecast_value), 4),
                "actual_kwh":   round(float(r.actual_value), 4),
                "error":        round(float(r.error), 4),
                "abs_error":    round(float(r.abs_error), 4),
            }
            for r in rows
        ],
    }


@app.get(
    "/data/stats",
    response_model=TrainingDataStatsResponse,
    summary="Статистика по данным",
    tags=["Данные"],
)
async def training_data_stats(db: Session = Depends(get_db)):
    total = db.query(func.count(TrainingData.id)).scalar() or 0
    first_date = last_date = None
    if total > 0:
        row_min = db.query(func.min(TrainingData.date)).scalar()
        row_max = db.query(func.max(TrainingData.date)).scalar()
        if row_min:
            first_date = row_min.isoformat() if hasattr(row_min, "isoformat") else str(row_min)
        if row_max:
            last_date  = row_max.isoformat() if hasattr(row_max, "isoformat") else str(row_max)
    return TrainingDataStatsResponse(
        total_rows=total,
        first_date=first_date,
        last_date=last_date,
        enough_for_predict=total >= MIN_POINTS_FOR_PREDICT,
        enough_for_backtest=total >= 200,
    )


@app.get("/anomalies", response_model=List[AnomalyEventResponse], summary="Аномалии", tags=["Данные"])
async def get_anomalies(
    from_time:     Optional[datetime] = Query(None),
    to_time:       Optional[datetime] = Query(None),
    anomalies_only: bool              = Query(False),
    limit:         int                = Query(100, ge=1, le=2000),
    db:            Session            = Depends(get_db),
):
    q = db.query(AnomalyEvent)
    if from_time:
        q = q.filter(AnomalyEvent.event_time >= from_time)
    if to_time:
        q = q.filter(AnomalyEvent.event_time <= to_time)
    if anomalies_only:
        q = q.filter(AnomalyEvent.is_anomaly == True)
    rows = q.order_by(AnomalyEvent.event_time.desc()).limit(limit).all()

    def fin(v):
        return (float(v) if np.isfinite(float(v)) else None) if v is not None else None

    return [
        AnomalyEventResponse(
            id=r.id, event_time=r.event_time, actual_value=r.actual_value,
            forecast_value=r.forecast_value, residual=r.residual,
            residual_threshold=fin(r.residual_threshold), stat_score=r.stat_score,
            rule_residual=r.rule_residual, rule_stat=r.rule_stat,
            is_anomaly=r.is_anomaly, reason=r.reason, created_at=r.created_at,
        )
        for r in rows
    ]


@app.get("/anomalies/stats", response_model=AnomalyStatsResponse, summary="Статистика аномалий", tags=["Данные"])
async def get_anomaly_stats(
    from_time: Optional[datetime] = Query(None),
    to_time:   Optional[datetime] = Query(None),
    db:        Session            = Depends(get_db),
):
    q = db.query(AnomalyEvent)
    if from_time:
        q = q.filter(AnomalyEvent.event_time >= from_time)
    if to_time:
        q = q.filter(AnomalyEvent.event_time <= to_time)
    total         = q.count()
    anomaly_count = q.filter(AnomalyEvent.is_anomaly == True).count()
    return AnomalyStatsResponse(
        from_time=from_time, to_time=to_time,
        total_events=total, anomaly_events=anomaly_count,
        anomaly_rate=float(anomaly_count / total) if total else 0.0,
    )


@app.post(
    "/upload-data",
    response_model=DataUploadResponse,
    summary="Загрузить данные потребления",
    tags=["Данные"],
)
async def upload_data(
    file:                    UploadFile = File(..., description="CSV: WsDateTime (или date/Datetime), consumption_kWh"),
    replace:                 bool       = Query(False, description="Если true — очистить таблицу перед вставкой"),
    skip_anomaly_detection:  bool       = Query(False, description="Если true — пропустить детекцию аномалий при загрузке (для быстрой массовой загрузки). Запустите POST /backtest/compute вручную после загрузки."),
    db:                      Session    = Depends(get_db),
):
    """
    Загрузка данных (15 мин). Принимает файл total_load_15min_full_kWh_new.csv.
    Поддерживаемые колонки: WsDateTime/date/Datetime + consumption_kWh.

    При загрузке большого датасета skip_anomaly_detection=true. 
    После завершения  POST /backtest/compute.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Нужен CSV файл")

    try:
        if replace:
            deleted = db.query(TrainingData).delete()
            db.query(AnomalyEvent).delete()
            db.commit()
            logger.info(f"Очищено {deleted} записей в training_data")

        contents = await file.read()
        df       = pd.read_csv(StringIO(contents.decode("utf-8")))

        value_col = next(
            (c for c in ("consumption_kWh", "consumption_kwh") if c in df.columns),
            None,
        )
        if value_col is None:
            raise HTTPException(status_code=400, detail="Нужна колонка consumption_kWh")

        for dc in ("WsDateTime", "date", "Datetime", "datetime"):
            if dc in df.columns:
                df["date"] = df[dc]
                break
        if "date" not in df.columns:
            raise HTTPException(status_code=400, detail="Нужна колонка WsDateTime, date или Datetime")

        df["date"] = _parse_upload_date(pd.Series(df["date"]))
        df         = df.dropna(subset=["date"])

        rows_added    = 0
        inserted_facts: list[tuple[datetime, float]] = []
        for _, row in df.iterrows():
            fact_time  = row["date"].to_pydatetime() if hasattr(row["date"], "to_pydatetime") else row["date"]
            fact_value = float(row[value_col])
            db.add(TrainingData(date=fact_time, usage_kwh=fact_value))
            inserted_facts.append((fact_time, fact_value))
            rows_added += 1
        db.commit()

        inserted_facts.sort(key=lambda x: x[0])
        if skip_anomaly_detection:
            logger.info(
                "skip_anomaly_detection=true: детекция аномалий пропущена. "
                "Запустите POST /backtest/compute для анализа после загрузки."
            )
        else:
            for fact_time, fact_value in inserted_facts:
                event = detect_anomaly_for_fact(db, fact_time=fact_time, fact_value=fact_value)
                record_anomaly_event(
                    is_anomaly=bool(event.is_anomaly),
                    residual=float(event.residual),
                    stat_score=event.stat_score,
                )

        total_rows = db.query(func.count(TrainingData.id)).scalar()
        record_data_upload(rows_added, success=True)
        update_training_data_count(total_rows or 0)

        # Прогноз по последним 96 точкам с учётом только что загруженных данных.
        # compute_and_save_forecast берёт актуальные данные из БД (уже включая новые),
        # сохраняет 4 committed-прогноза в forecast_cache и возвращает их значения.
        forecast_result: dict | None = None
        forecast_skipped: str | None = None
        if model is None or scaler is None:
            forecast_skipped = "модель не загружена"
        elif (total_rows or 0) < MIN_POINTS_FOR_PREDICT:
            forecast_skipped = (
                f"недостаточно данных: {total_rows} строк, нужно минимум {MIN_POINTS_FOR_PREDICT}"
            )
        else:
            try:
                forecast_result = compute_and_save_forecast(db)
            except Exception as exc:
                forecast_skipped = str(exc)
                logger.warning("Прогноз после upload-data не построен: %s", exc)

        last_fact_date = inserted_facts[-1][0].isoformat() if inserted_facts else None

        forecast_times = None
        if forecast_result:
            last_fact_dt = datetime.fromisoformat(forecast_result["last_fact_date"])
            forecast_times = [
                (last_fact_dt + timedelta(minutes=15 * i)).isoformat()
                for i in range(1, 5)
            ]

        # Авто-бэктест: накапливаем prediction-историю в backtest_cache.
        # Пропускаем при массовой загрузке — пользователь запустит вручную.
        if (
            not skip_anomaly_detection
            and model is not None
            and scaler is not None
            and (total_rows or 0) >= MIN_POINTS_FOR_PREDICT + 2
        ):
            try:
                bt_steps = max(5, min(rows_added, BACKTEST_AUTO_STEPS))
                compute_and_save_backtest(db, steps=bt_steps)
            except Exception as exc:
                logger.warning("Авто-бэктест после upload-data не выполнен: %s", exc)

        return DataUploadResponse(
            message="Данные загружены",
            rows_uploaded=rows_added,
            total_rows=total_rows,
            forecast=forecast_result["values"] if forecast_result else None,
            forecast_times=forecast_times,
            last_fact_date=last_fact_date,
            forecast_skipped_reason=forecast_skipped,
        )
    except HTTPException:
        raise
    except Exception as e:
        record_data_upload(0, success=False)
        logger.exception("Ошибка загрузки данных")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Главная", tags=["Система"])
async def root():
    return {
        "сервис":      "Total Load LSTM (Direct Multi-Step, 15 мин)",
        "версия":      "5.0.0",
        "горизонт":    "4 × 15 мин = 1 ч (+15/+30/+45/+60 мин)",
        "вход_модели": f"{SEQ_LEN} точек × {N_FEATURES} признаков",
        "документация": "/docs",
        "состояние":   "/health",
        "обновить_модель": "/model/reload",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
