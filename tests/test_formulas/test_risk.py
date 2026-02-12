"""Tests for risk metric formulas."""

import numpy as np
import pandas as pd
import pytest
from etf_analyzer.formulas.risk import (
    max_drawdown,
    max_drawdown_duration,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    alpha_beta,
    downside_risk,
    return_drawdown_ratio,
)


class TestMaxDrawdown:
    def test_simple_drawdown(self):
        nav = pd.Series([1.0, 1.2, 1.1, 0.9, 1.0])
        result = max_drawdown(nav)
        assert result == pytest.approx(-0.25, abs=1e-6)

    def test_no_drawdown(self):
        nav = pd.Series([1.0, 1.1, 1.2, 1.3])
        result = max_drawdown(nav)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_single_value(self):
        nav = pd.Series([1.0])
        result = max_drawdown(nav)
        assert result == pytest.approx(0.0, abs=1e-6)


class TestMaxDrawdownDuration:
    def test_basic_duration(self):
        nav = pd.Series(
            [1.0, 1.1, 1.2, 1.0, 0.9, 1.0, 1.1, 1.2],
            index=pd.bdate_range("2024-01-02", periods=8),
        )
        duration = max_drawdown_duration(nav)
        assert duration >= 3


class TestAnnualizedVolatility:
    def test_known_volatility(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 252))
        vol = annualized_volatility(returns)
        assert vol == pytest.approx(0.01 * np.sqrt(252), abs=0.02)


class TestSharpeRatio:
    def test_positive_sharpe(self, sample_daily_returns):
        sr = sharpe_ratio(sample_daily_returns, risk_free_rate=0.02)
        assert sr > 0

    def test_zero_volatility_returns_zero(self):
        flat_returns = pd.Series([0.001] * 252)
        sr = sharpe_ratio(flat_returns, risk_free_rate=0.0)
        assert isinstance(sr, float)


class TestAlphaBeta:
    def test_alpha_beta_returns_tuple(
        self, sample_daily_returns, sample_benchmark_returns
    ):
        a, b = alpha_beta(sample_daily_returns, sample_benchmark_returns)
        assert isinstance(a, float)
        assert isinstance(b, float)


class TestCalmarRatio:
    def test_positive_calmar(self, sample_nav_series):
        returns = sample_nav_series.pct_change().dropna()
        result = calmar_ratio(returns, sample_nav_series)
        assert isinstance(result, float)


class TestReturnDrawdownRatio:
    def test_basic_ratio(self):
        result = return_drawdown_ratio(annual_return=0.1, max_dd=-0.05)
        assert result == pytest.approx(2.0, abs=1e-6)

    def test_zero_drawdown(self):
        result = return_drawdown_ratio(annual_return=0.1, max_dd=0.0)
        assert result == float("inf")
