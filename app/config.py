"""
Конфигурация приложения.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Настройки приложения."""

    # База данных
    database_url: str = "postgresql://postgres:postgres@postgres:5432/mlservice"

    # LSTM-модель (Total Load 15 мин, Direct Multi-Step, 4 выхода)
    model_path:  str = "models/best_lstm_total_load_multistep.pt"
    params_path: str = "models/best_lstm_total_load_multistep_params.pkl"
    scaler_path: str = "models/lstm_total_load_15min_scaler.joblib"

    # Архитектурные параметры Total Load LSTM (Direct Multi-Step)
    target_column: str = "consumption_kWh"
    seq_len:       int  = 96   # входная последовательность (24 ч при шаге 15 мин)
    horizon:       int  = 4    # прогноз HORIZON × 15 мин = 1 ч вперёд (4 выхода)
    n_features:    int  = 6    # consumption + 5 календарных

    # Детекция аномалий
    # anomaly_window_size: окно калибровки порога.
    # 672 = 7 дней × 96 точек/день.  Порог учитывает полный недельный цикл
    # (ночь/день, будни/выходные), что исключает ложные срабатывания на
    # суточных и недельных переходах.
    anomaly_window_size:     int   = 672   # 7 дней
    residual_sigma_k:        float = 3.0
    mad_k:                   float = 6.0
    # Минимум точек для начала детекции.  672 = полная неделя: не детектируем
    # аномалии пока не накоплен репрезентативный контекст.
    min_points_for_anomaly:  int   = 672
    # Минимальный абсолютный порог residual (кВт·ч).
    # Итоговый порог = max(adaptive, floor).
    # Защищает от ложных срабатываний когда adaptive_threshold слишком мал.
    anomaly_min_residual:    float = 0.15
    # Минимальный относительный порог (доля от actual_value).
    # Итоговый порог = max(adaptive, abs_floor, rel_floor * actual).
    # 0.15 → не флагировать если ошибка < 15% от факта.
    # Для LSTM ошибка 5–10% — рабочая норма; аномалия — это >15%.
    anomaly_min_residual_pct: float = 0.15

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Получение настроек приложения (с кэшированием)."""
    return Settings()
