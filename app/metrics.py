"""
Prometheus метрики для мониторинга ML сервиса.
"""
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps

# Метрики прогнозирования
PREDICTIONS_TOTAL = Counter(
    'model_predictions_total',
    'Общее количество выполненных прогнозов',
    ['status']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Время выполнения прогноза в секундах',
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

PREDICTION_VALUE = Histogram(
    'model_prediction_value',
    'Распределение прогнозируемых значений',
    buckets=[0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200]
)

MODEL_MAE = Gauge('model_mae', 'Средняя абсолютная ошибка модели (масштабированная)')
MODEL_RMSE = Gauge('model_rmse', 'Среднеквадратичная ошибка модели (масштабированная)')
MODEL_R2 = Gauge('model_r2', 'Коэффициент детерминации R² модели')
MODEL_MSE = Gauge('model_mse', 'Среднеквадратичная ошибка модели MSE (масштабированная)')

# Метрики качества модели по горизонту прогноза (step=1..4 → +15,+30,+45,+60 мин)
MODEL_STEP_MAE = Gauge(
    'model_step_mae',
    'MAE по шагу прогноза (кВт·ч)',
    ['step']
)

MODEL_STEP_RMSE = Gauge(
    'model_step_rmse',
    'RMSE по шагу прогноза (кВт·ч)',
    ['step']
)

MODEL_STEP_MAPE = Gauge(
    'model_step_mape',
    'MAPE по шагу прогноза (%)',
    ['step']
)

# Метрики данных
TRAINING_DATA_COUNT = Gauge(
    'training_data_count',
    'Количество записей в таблице training_data'
)

LAST_MODEL_LOAD_TIMESTAMP = Gauge(
    'last_model_load_timestamp',
    'Unix-время последней загрузки/перезагрузки модели'
)

# Метрики загрузки данных
DATA_UPLOADS_TOTAL = Counter(
    'data_uploads_total',
    'Общее количество загрузок данных',
    ['status']
)

UPLOADED_ROWS = Counter(
    'uploaded_rows_total',
    'Общее количество загруженных строк'
)

# Метрики аномалий
ANOMALY_EVENTS_TOTAL = Counter(
    'anomaly_events_total',
    'Общее количество событий детекции аномалий',
    ['status']  # anomaly | normal
)

ANOMALY_RESIDUAL = Histogram(
    'anomaly_residual_value',
    'Распределение абсолютной ошибки |fact-forecast| для проверенных фактов',
    buckets=[0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
)

ANOMALY_STAT_SCORE = Histogram(
    'anomaly_stat_score',
    'Распределение robust stat-score по проверенным фактам',
    buckets=[0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
)

# Информация о модели
MODEL_INFO = Info(
    'model',
    'Информация о текущей модели'
)


def update_model_metrics(metrics: dict) -> None:
    """Обновление метрик качества модели."""
    if 'mae' in metrics:
        MODEL_MAE.set(metrics['mae'])
    if 'rmse' in metrics:
        MODEL_RMSE.set(metrics['rmse'])
    if 'r2' in metrics:
        MODEL_R2.set(metrics['r2'])
    if 'mse' in metrics:
        MODEL_MSE.set(metrics['mse'])


def update_training_data_count(count: int) -> None:
    """Обновление счётчика записей в training_data."""
    TRAINING_DATA_COUNT.set(count)


def update_step_metrics(step_errors: dict) -> None:
    """Обновление per-step метрик после backtest.

    step_errors: {step_num: [(abs_err, sq_err, pct_err), ...]}
    """
    for step_num, errs in step_errors.items():
        if not errs:
            continue
        n = len(errs)
        mae  = sum(e[0] for e in errs) / n
        rmse = (sum(e[1] for e in errs) / n) ** 0.5
        mape = sum(e[2] for e in errs) / n
        label = str(step_num)
        MODEL_STEP_MAE.labels(step=label).set(mae)
        MODEL_STEP_RMSE.labels(step=label).set(rmse)
        MODEL_STEP_MAPE.labels(step=label).set(mape)


def update_model_info(info: dict) -> None:
    """Обновление информации о модели."""
    MODEL_INFO.info({k: str(v) for k, v in info.items()})


def record_prediction(latency: float, value: float, success: bool = True) -> None:
    """Запись метрик прогнозирования."""
    status = 'success' if success else 'error'
    PREDICTIONS_TOTAL.labels(status=status).inc()
    PREDICTION_LATENCY.observe(latency)
    if success:
        PREDICTION_VALUE.observe(value)


def record_model_load() -> None:
    """Запись факта загрузки/перезагрузки модели."""
    LAST_MODEL_LOAD_TIMESTAMP.set(time.time())


def record_data_upload(rows: int, success: bool = True) -> None:
    """Запись метрик загрузки данных."""
    status = 'success' if success else 'error'
    DATA_UPLOADS_TOTAL.labels(status=status).inc()
    if success:
        UPLOADED_ROWS.inc(rows)


def record_anomaly_event(is_anomaly: bool, residual: float, stat_score: float | None) -> None:
    """Запись метрик детекции аномалий."""
    ANOMALY_EVENTS_TOTAL.labels(status='anomaly' if is_anomaly else 'normal').inc()
    ANOMALY_RESIDUAL.observe(max(float(residual), 0.0))
    if stat_score is not None:
        ANOMALY_STAT_SCORE.observe(max(float(stat_score), 0.0))


def timed_prediction(func):
    """Декоратор для измерения времени прогнозирования."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            latency = time.time() - start_time
            # Извлечение значения прогноза если доступно
            pred_value = getattr(result, 'prediction', 0)
            record_prediction(latency, pred_value, success=True)
            return result
        except Exception as e:
            latency = time.time() - start_time
            record_prediction(latency, 0, success=False)
            raise e
    return wrapper
