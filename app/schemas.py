"""
Pydantic схемы для API прогнозирования потребления электроэнергии
(Total Load 15 мин, LSTM Direct Multi-Step, 4 выхода).
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class DataPoint(BaseModel):
    """Одна точка данных: дата и потребление за интервал (кВт·ч)."""
    date: Optional[str] = Field(None, description="Дата и время (интервал 15 минут)")
    consumption_kWh: float = Field(..., description="Потребление за интервал, кВт·ч")


class PredictionRequest(BaseModel):
    """Запрос на прогноз: история потребления (минимум 96 точек)."""
    data: List[DataPoint] = Field(
        ...,
        min_length=96,
        description="Исторические точки (date, consumption_kWh). Минимум 96 (SEQ_LEN = 24 ч) для LSTM-прогноза.",
    )


class PredictionResponse(BaseModel):
    """Ответ: прогноз на 4 шага вперёд (+15/+30/+45/+60 мин), кВт·ч. Direct Multi-Step LSTM."""
    forecast: Dict[str, float] = Field(
        ...,
        description="Прогноз по горизонтам: {'+15min': float, '+30min': float, '+45min': float, '+60min': float}",
    )
    forecast_times: List[str] = Field(
        ...,
        description="Временны́е метки прогнозов (ISO 8601), 4 значения",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Время ответа (UTC)")


class DataUploadResponse(BaseModel):
    """Ответ после загрузки данных."""
    message: str = Field(description="Сообщение о результате")
    rows_uploaded: int = Field(description="Количество загруженных строк")
    total_rows: int = Field(description="Общее количество строк в базе данных")
    forecast: Optional[Dict[str, float]] = Field(
        None,
        description="Прогноз на +15/+30/+45/+60 мин от последней загруженной точки (кВт·ч)",
    )
    forecast_times: Optional[List[str]] = Field(
        None,
        description="Временны́е метки прогнозов (ISO 8601), 4 значения",
    )
    last_fact_date: Optional[str] = Field(
        None,
        description="Дата последней загруженной точки (ISO 8601)",
    )
    forecast_skipped_reason: Optional[str] = Field(
        None,
        description="Причина, по которой прогноз не был построен (если модель не загружена или мало данных)",
    )


class TrainingDataStatsResponse(BaseModel):
    """Статистика по данным в training_data."""
    total_rows: int = Field(description="Количество записей в training_data")
    first_date: Optional[str] = Field(None, description="Дата первой записи (ISO)")
    last_date: Optional[str] = Field(None, description="Дата последней записи (ISO)")
    enough_for_predict: bool = Field(description="Достаточно ли данных для прогноза (нужно минимум 96)")
    enough_for_backtest: bool = Field(description="Достаточно ли данных для бэктеста (рекомендуется ≥200)")


class AnomalyEventResponse(BaseModel):
    """Событие детекции аномалии."""
    id: int
    event_time: datetime
    actual_value: float
    forecast_value: float
    residual: float
    residual_threshold: Optional[float] = None
    stat_score: Optional[float] = None
    rule_residual: bool
    rule_stat: bool
    is_anomaly: bool
    reason: Optional[str] = None
    created_at: datetime


class AnomalyStatsResponse(BaseModel):
    """Сводка по событиям аномалий за период."""
    from_time: Optional[datetime] = None
    to_time: Optional[datetime] = None
    total_events: int
    anomaly_events: int
    anomaly_rate: float


class HealthResponse(BaseModel):
    """Ответ проверки состояния сервиса."""
    status: str = Field(description="Статус сервиса")
    model_loaded: bool = Field(description="Загружена ли модель")
    database_connected: bool = Field(description="Подключена ли база данных")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Время проверки")


class MetricsResponse(BaseModel):
    """Метрики модели (из БД)."""
    mae: Optional[float] = Field(None, description="Средняя абсолютная ошибка (MAE)")
    rmse: Optional[float] = Field(None, description="Среднеквадратичная ошибка (RMSE)")
    r2: Optional[float] = Field(None, description="Коэффициент детерминации R²")
    mse: Optional[float] = Field(None, description="MSE")
    last_updated: Optional[datetime] = Field(None, description="Время последнего обновления метрик")
    training_samples: Optional[int] = Field(None, description="Количество обучающих примеров")
