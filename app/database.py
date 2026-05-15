"""
Модуль работы с базой данных.
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.config import get_settings

settings = get_settings()

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class TrainingData(Base):
    """Таблица для хранения данных потребления (Total Load, 15 мин)."""
    __tablename__ = "training_data"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, index=True, comment="Дата и время измерения (15-мин интервал)")
    usage_kwh = Column(Float, comment="Потребление электроэнергии в кВт·ч (consumption_kWh)")
    created_at = Column(DateTime, default=datetime.utcnow, comment="Дата создания записи")



class ForecastCache(Base):
    """История прогнозов вперёд. Каждый запуск планировщика добавляет 4 строки (не перезаписывает).
    is_committed=True означает: прогноз был сохранён ДО того, как факт стал известен.
    Используется в detect_anomaly_for_fact для честного сравнения «committed forecast vs факт».
    """
    __tablename__ = "forecast_cache"

    id = Column(Integer, primary_key=True, index=True)
    computed_at = Column(DateTime, nullable=False, index=True, comment="Когда выполнен расчёт")
    forecast_time = Column(DateTime, nullable=False, index=True, comment="Метка времени прогноза")
    value_kwh = Column(Float, nullable=False, comment="Прогноз нагрузки (кВт·ч на 15 мин)")
    is_committed = Column(Boolean, nullable=False, default=True,
                          comment="True = прогноз сохранён планировщиком до прихода факта")


class AnomalyEvent(Base):
    """События детекции аномалий по фактическим данным."""
    __tablename__ = "anomaly_events"

    id = Column(Integer, primary_key=True, index=True)
    event_time = Column(DateTime, nullable=False, index=True, comment="Время фактического измерения")
    actual_value = Column(Float, nullable=False, comment="Фактическое значение, кВт·ч")
    forecast_value = Column(Float, nullable=False, comment="Прогноз для этого времени, кВт·ч")
    residual = Column(Float, nullable=False, comment="Абсолютная ошибка |fact - forecast|")
    residual_threshold = Column(Float, nullable=False, comment="Порог residual-rule")
    stat_score = Column(Float, nullable=True, comment="Робастный stat-score (MAD z-score)")
    rule_residual = Column(Boolean, nullable=False, default=False, comment="Сработало residual-rule")
    rule_stat = Column(Boolean, nullable=False, default=False, comment="Сработало stat-rule")
    is_anomaly = Column(Boolean, nullable=False, default=False, index=True, comment="Итоговый флаг аномалии")
    reason = Column(String, nullable=True, comment="Причина/комментарий")
    created_at = Column(DateTime, default=datetime.utcnow, comment="Дата создания записи")


class BacktestCache(Base):
    """Кэш сравнения прогноза и факта на исторических точках (backtest)."""
    __tablename__ = "backtest_cache"

    id = Column(Integer, primary_key=True, index=True)
    computed_at = Column(DateTime, nullable=False, index=True, comment="Когда выполнен расчёт backtest")
    point_time = Column(DateTime, nullable=False, index=True, comment="Время фактической точки")
    step_num = Column(Integer, nullable=False, default=1, comment="Шаг прогноза (1=+15мин, 2=+30мин, 3=+45мин, 4=+60мин)")
    forecast_value = Column(Float, nullable=False, comment="Прогноз, кВт·ч")
    actual_value = Column(Float, nullable=False, comment="Факт, кВт·ч")
    error = Column(Float, nullable=False, comment="Ошибка fact - forecast")
    abs_error = Column(Float, nullable=False, comment="Абсолютная ошибка")
    created_at = Column(DateTime, default=datetime.utcnow, comment="Дата создания записи")


def get_db():
    """Получение сессии базы данных."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Инициализация таблиц базы данных."""
    Base.metadata.create_all(bind=engine)
