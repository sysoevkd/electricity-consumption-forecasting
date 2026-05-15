"""
Юнит-тесты предобработки LSTM (app/preprocessing.py): Total Load, 15 мин.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from app.preprocessing import (
    TARGET_COL,
    SEQ_LEN,
    N_FEATURES,
    HORIZON,
    build_lstm_features,
    prepare_lstm_input,
    get_feature_columns,
)


def _synthetic_series(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range(start="2024-01-01 00:00:00", periods=n, freq="15min")
    np.random.seed(0)
    v = 3.0 + np.cumsum(np.random.randn(n) * 0.05)
    return pd.DataFrame({"date": dates, TARGET_COL: np.clip(v, 0.5, 20.0)})


class TestBuildLstmFeatures:
    def test_shape_and_columns(self):
        df = _synthetic_series(100)
        X = build_lstm_features(df)
        assert X.shape == (100, N_FEATURES)
        assert np.isfinite(X).all()

    def test_aliases_usage_kwh(self):
        df = _synthetic_series(50)
        df2 = df.rename(columns={TARGET_COL: "usage_kwh"})
        X1 = build_lstm_features(df)
        X2 = build_lstm_features(df2)
        np.testing.assert_allclose(X1, X2)

    def test_hour_weekend_in_range(self):
        df = _synthetic_series(200)
        X = build_lstm_features(df)
        # hour_sin, hour_cos, dow_sin, dow_cos roughly in [-1, 1]
        assert X[:, 1:5].min() >= -1.01
        assert X[:, 1:5].max() <= 1.01
        assert set(np.unique(X[:, 5])).issubset({0.0, 1.0})


class TestPrepareLstmInput:
    def test_output_shape(self):
        df = _synthetic_series(SEQ_LEN + 10)
        scaler = MinMaxScaler()
        scaler.fit(build_lstm_features(df))
        arr = prepare_lstm_input(df, scaler, seq_len=SEQ_LEN)
        assert arr.shape == (1, SEQ_LEN, N_FEATURES)
        assert arr.dtype == np.float32

    def test_raises_if_too_short(self):
        df = _synthetic_series(50)
        scaler = MinMaxScaler()
        scaler.fit(build_lstm_features(_synthetic_series(SEQ_LEN)))
        with pytest.raises(ValueError, match="Недостаточно данных"):
            prepare_lstm_input(df, scaler, seq_len=SEQ_LEN)


class TestFeatureColumnsDoc:
    def test_count_matches_n_features(self):
        cols = get_feature_columns()
        assert len(cols) == N_FEATURES == 6


class TestHorizonConstant:
    def test_horizon_is_four_steps(self):
        assert HORIZON == 4
