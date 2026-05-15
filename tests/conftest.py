"""
Фикстуры для тестов Total Load LSTM (15 мин, Direct Multi-Step).

Интеграционные тесты ждут API по API_BASE_URL (по умолчанию http://localhost:8000).
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
# Минимум точек для POST /predict и обучения окна LSTM
MIN_LSTM_POINTS = 96

# CSV с колонками consumption_kWh (или алиасы) для интеграционных тестов
DEFAULT_DATA_CSV = ROOT / "total_load_15min_full_kWh_new.csv"
FALLBACK_DATA_CSV = ROOT / "data_upload_test.csv"


@pytest.fixture(scope="session")
def api_base_url():
    return BASE_URL.rstrip("/")


@pytest.fixture(scope="session")
def total_load_csv_path():
    """Файл с рядом Total Load; для прогноза нужно ≥96 последних точек."""
    if DEFAULT_DATA_CSV.exists():
        return DEFAULT_DATA_CSV
    if FALLBACK_DATA_CSV.exists():
        return FALLBACK_DATA_CSV
    pytest.skip(
        f"Нет CSV с данными: положите {DEFAULT_DATA_CSV.name} или {FALLBACK_DATA_CSV.name} в корень проекта."
    )


@pytest.fixture(scope="session")
def lstm_predict_payload(total_load_csv_path):
    """Тело POST /predict: минимум 96 точек (date + consumption_kWh)."""
    import pandas as pd

    df = pd.read_csv(total_load_csv_path)
    if "consumption_kWh" not in df.columns:
        for c in ("usage_kwh", "Usage_kWh"):
            if c in df.columns:
                df["consumption_kWh"] = df[c]
                break
    date_col = "date" if "date" in df.columns else "WsDateTime"
    df["date"] = pd.to_datetime(df[date_col], format="mixed", errors="coerce")
    df = df.dropna(subset=["date", "consumption_kWh"]).tail(MIN_LSTM_POINTS)
    if len(df) < MIN_LSTM_POINTS:
        pytest.skip(f"В {total_load_csv_path.name} меньше {MIN_LSTM_POINTS} строк после очистки.")
    data = [
        {"date": row["date"].isoformat(), "consumption_kWh": float(row["consumption_kWh"])}
        for _, row in df.iterrows()
    ]
    return {"data": data}


@pytest.fixture
def small_upload_csv(total_load_csv_path):
    """Небольшой CSV для POST /upload-data (первые ~120 строк)."""
    import pandas as pd

    df = pd.read_csv(total_load_csv_path)
    if "consumption_kWh" not in df.columns:
        for c in ("usage_kwh", "Usage_kWh"):
            if c in df.columns:
                df["consumption_kWh"] = df[c]
                break
    date_col = "date" if "date" in df.columns else "WsDateTime"
    df["date"] = pd.to_datetime(df[date_col], format="mixed", errors="coerce")
    df = df[["date", "consumption_kWh"]].dropna().head(120)
    return df.to_csv(index=False)
