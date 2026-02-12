"""Tests for technical analysis formulas."""

import numpy as np
import pandas as pd
import pytest
from etf_analyzer.formulas.technical import (
    moving_average,
    is_ma_uptrend,
    daily_change_pct,
    drawdown_from_peak,
    rolling_volatility,
)


class TestMovingAverage:
    def test_5_day_ma(self):
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        ma = moving_average(prices, window=5)
        assert ma.iloc[-1] == pytest.approx(5.0, abs=1e-9)
        assert ma.isna().sum() == 4

    def test_window_larger_than_series(self):
        prices = pd.Series([1.0, 2.0])
        ma = moving_average(prices, window=5)
        assert ma.isna().all()


class TestMaUptrend:
    def test_uptrend(self):
        prices = pd.Series(range(1, 70), dtype=float)
        assert is_ma_uptrend(prices, window=60) is True

    def test_downtrend(self):
        prices = pd.Series(range(70, 1, -1), dtype=float)
        assert is_ma_uptrend(prices, window=60) is False


class TestDailyChangePct:
    def test_basic_change(self):
        prices = pd.Series([100.0, 103.0, 100.0])
        changes = daily_change_pct(prices)
        assert changes.iloc[0] == pytest.approx(0.03, abs=1e-6)
        assert changes.iloc[1] == pytest.approx(-0.0291, abs=1e-3)


class TestDrawdownFromPeak:
    def test_basic_drawdown(self):
        nav = pd.Series([1.0, 1.2, 1.1, 0.9, 1.0])
        dd = drawdown_from_peak(nav)
        assert dd.iloc[3] == pytest.approx(-0.25, abs=1e-6)
        assert dd.iloc[1] == pytest.approx(0.0, abs=1e-6)


class TestRollingVolatility:
    def test_rolling_vol_length(self):
        returns = pd.Series(np.random.normal(0, 0.01, 100))
        vol = rolling_volatility(returns, window=20)
        assert len(vol) == len(returns)
        assert vol.isna().sum() == 19
