"""
Предобработка данных для LSTM-модели (Total Load, 15 мин).

Вход модели: последовательность SEQ_LEN точек, каждая точка — вектор из N_FEATURES признаков:
  [consumption_kWh, hour_sin, hour_cos, dow_sin, dow_cos, is_weekend]
Нормализация: MinMaxScaler, обученный на train-части данных (передаётся явно).
Горизонт прогноза: HORIZON шагов × 15 мин = 1 час вперёд.
"""
import pandas as pd
import numpy as np
from typing import Optional

TARGET_COL = "consumption_kWh"
N_FEATURES = 6    # consumption + hour_sin + hour_cos + dow_sin + dow_cos + is_weekend
SEQ_LEN    = 96   # 96 × 15 мин = 24 ч истории
HORIZON    = 4    # предсказываем на HORIZON шагов вперёд (4 × 15 мин = 1 ч)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализует имена колонок (date, consumption_kWh)."""
    df = df.copy()
    if TARGET_COL not in df.columns:
        for alias in ("usage_kwh", "Usage_kWh", "usage_kWh"):
            if alias in df.columns:
                df[TARGET_COL] = df[alias]
                break
    if "date" not in df.columns:
        for alias in ("WsDateTime", "Datetime", "datetime"):
            if alias in df.columns:
                df["date"] = df[alias]
                break
    return df


def _parse_date(ser: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(ser):
        return pd.to_datetime(ser)
    return pd.to_datetime(ser, format="mixed", errors="coerce")


def build_lstm_features(df: pd.DataFrame) -> np.ndarray:
    """
    Строит матрицу признаков (n, N_FEATURES) из df с колонками date и consumption_kWh.
    Порядок признаков: [consumption_kWh, hour_sin, hour_cos, dow_sin, dow_cos, is_weekend]

    Args:
        df: DataFrame с колонками date (или WsDateTime) и consumption_kWh (или usage_kwh)

    Returns:
        numpy array формы (n, 6), данные не нормализованы 
    """
    df = _normalize_df(df)
    df["date"] = _parse_date(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Не найдена колонка {TARGET_COL} или её алиасы")

    dt = pd.to_datetime(df["date"])
    consumption = df[TARGET_COL].astype(float).values

    hour = dt.dt.hour + dt.dt.minute / 60.0
    dayofweek = dt.dt.dayofweek.astype(float)

    features = np.column_stack([
        consumption,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dayofweek / 7),
        np.cos(2 * np.pi * dayofweek / 7),
        (dt.dt.dayofweek >= 5).astype(float),
    ]).astype(np.float32)  # (n, 6)

    return features


def prepare_lstm_input(
    df: pd.DataFrame,
    scaler,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """
    Подготавливает вход для LSTM: последние seq_len точек → (1, seq_len, N_FEATURES).

    Args:
        df:      DataFrame с историей (минимум seq_len строк), колонки date + consumption_kWh
        scaler:  обученный MinMaxScaler
        seq_len: длина входной последовательности (default 96)

    Returns:
        numpy array формы (1, seq_len, N_FEATURES), dtype float32
    """
    features = build_lstm_features(df)
    if len(features) < seq_len:
        raise ValueError(
            f"Недостаточно данных: {len(features)} точек, нужно минимум {seq_len}."
        )
    seq = features[-seq_len:]              # (seq_len, 6)
    seq_scaled = scaler.transform(seq)     # (seq_len, 6)
    return seq_scaled[np.newaxis, :, :]   # (1, seq_len, 6)


def get_feature_columns() -> list:
    """Список имён признаков в порядке следования """
    return ["consumption_kWh", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
