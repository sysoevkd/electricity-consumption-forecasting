"""
Загрузка и инференс LSTM-модели (PyTorch) для прогноза потребления Total Load.

Архитектура: LSTMModelOptuna — Direct Multi-Step (6 признаков × 96 шагов → 4 значения).
Признаки: consumption_kWh + hour_sin + hour_cos + dow_sin + dow_cos + is_weekend.
Выход: вектор [+15 мин, +30 мин, +45 мин, +60 мин] в кВт·ч.
"""
import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import joblib

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch не установлен. Модель LSTM недоступна.")


def _build_lstm_class():
    """Создаёт класс LSTMModelOptuna"""
    import torch.nn as nn

    class LSTMModelOptuna(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                     dropout: float, out_steps: int = 4):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, out_steps)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])  # (batch, out_steps)

    return LSTMModelOptuna


def load_model(model_path: str, params_path: Optional[str] = None) -> Any:
    """
    Загружает LSTMModelOptuna из .pt файла (state_dict).

    Гиперпараметры читаются из params_path (.pkl), сохранённого ноутбуком:
      {hidden_size, num_layers, dropout, out_steps, look_back, n_features, ...}

    Returns:
        nn.Module в режиме eval() на CPU
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch не установлен."
        )

    import torch

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {path}")

    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    params: dict = {}
    if params_path:
        p = Path(params_path)
        if p.exists():
            params = joblib.load(p)
            logger.info(f"Гиперпараметры из {p}: {params}")

    hidden_size = int(params.get("hidden_size", 160))
    num_layers  = int(params.get("num_layers", 2))
    dropout     = float(params.get("dropout", 0.2))
    out_steps   = int(params.get("out_steps", 4))
    n_features  = int(params.get("n_features", 6))

    LSTMClass = _build_lstm_class()
    model = LSTMClass(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        out_steps=out_steps,
    )
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(
        f"LSTMModelOptuna загружена: hidden={hidden_size}, layers={num_layers}, "
        f"dropout={dropout}, out_steps={out_steps}, n_features={n_features}"
    )
    return model


def load_scaler(scaler_path: str) -> Any:
    """Загружает MinMaxScaler из .joblib файла."""
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Файл scaler не найден: {path}. "
            "Скопируйте lstm_total_load_15min_scaler.joblib в models/."
        )
    scaler = joblib.load(path)
    logger.info(f"Scaler загружен из {path}")
    return scaler


def predict_multi(model: Any, X_np: np.ndarray, scaler: Any, n_features: int = 6) -> List[float]:
    """
    Выполняет инференс LSTMModelOptuna — Direct Multi-Step.

    Args:
        model:      LSTMModelOptuna в eval-режиме
        X_np:       numpy array формы (1, seq_len, n_features)
        scaler:     MinMaxScaler, обученный на тренировочных данных (все 6 колонок)
        n_features: число признаков (default 6)

    Returns:
        Список из 4 денормализованных значений [+15 мин, +30 мин, +45 мин, +60 мин], кВт·ч
    """
    import torch

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        out = model(X_tensor)  # (1, 4)
    preds_scaled = out[0].numpy()  # (4,)

    # Обратное масштабирование каждого шага через колонку 0 скейлера
    result = []
    for val in preds_scaled:
        dummy = np.zeros((1, n_features), dtype=np.float32)
        dummy[0, 0] = float(val)
        result.append(float(scaler.inverse_transform(dummy)[0, 0]))
    return result
