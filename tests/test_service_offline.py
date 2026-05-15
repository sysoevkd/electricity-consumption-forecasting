"""
Офлайн-проверка: загрузка LSTM, scaler и один проход prepare_lstm_input + predict_multi.

Запуск: pytest tests/test_service_offline.py -v
Требуется: models/best_lstm_total_load_multistep.pt, models/lstm_total_load_15min_scaler.joblib
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_offline_lstm_load_and_prepare():
    from app.config import get_settings
    from app.model import load_model, load_scaler, predict_multi
    from app.preprocessing import TARGET_COL, prepare_lstm_input, SEQ_LEN, N_FEATURES

    settings = get_settings()
    model_path = Path(settings.model_path)
    scaler_path = Path(settings.scaler_path)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not scaler_path.is_absolute():
        scaler_path = ROOT / scaler_path

    if not model_path.exists() or not scaler_path.exists():
        pytest.skip("Нет файлов модели или scaler в models/")

    model = load_model(str(model_path), str(Path(settings.params_path)) if (ROOT / settings.params_path).exists() else None)
    scaler = load_scaler(str(scaler_path))

    csv = ROOT / "data_upload_test.csv"
    if not csv.exists():
        csv = ROOT / "total_load_15min_full_kWh_new.csv"
    if not csv.exists():
        pytest.skip("Нет CSV с данными для окна LSTM")

    df = pd.read_csv(csv)
    if "consumption_kWh" not in df.columns:
        pytest.skip("В CSV нет consumption_kWh")
    dc = "date" if "date" in df.columns else "WsDateTime"
    df["date"] = pd.to_datetime(df[dc], format="mixed", errors="coerce")
    df = df.dropna(subset=["date", "consumption_kWh"]).tail(SEQ_LEN)

    if len(df) < SEQ_LEN:
        pytest.skip(f"Нужно минимум {SEQ_LEN} строк в CSV")

    X = prepare_lstm_input(df, scaler)
    assert X.shape == (1, SEQ_LEN, N_FEATURES)

    preds = predict_multi(model, X, scaler, n_features=N_FEATURES)
    assert len(preds) == 4
    assert all(np.isfinite(preds))
    assert all(p >= 0 for p in preds)
