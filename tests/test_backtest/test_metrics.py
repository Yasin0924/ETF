"""Tests for backtest performance metrics."""

import pandas as pd
import pytest
from etf_analyzer.backtest.metrics import calculate_backtest_metrics


@pytest.fixture
def equity_curve():
    dates = pd.bdate_range("2024-01-02", periods=252)
    values = [100000 * (1 + 0.0003) ** i for i in range(252)]
    return [
        {"date": d.date(), "total_value": v, "cash": 5000}
        for d, v in zip(dates, values)
    ]


@pytest.fixture
def trade_log():
    return [
        {"date": "2024-01-02", "type": "buy", "amount": 50000},
        {"date": "2024-06-01", "type": "sell", "net_amount": 55000},
        {"date": "2024-06-15", "type": "buy", "amount": 55000},
        {"date": "2024-12-01", "type": "sell", "net_amount": 52000},
    ]


class TestBacktestMetrics:
    def test_returns_dict(self, equity_curve, trade_log):
        metrics = calculate_backtest_metrics(
            equity_curve, trade_log, initial_capital=100000
        )
        assert isinstance(metrics, dict)

    def test_has_required_metrics(self, equity_curve, trade_log):
        metrics = calculate_backtest_metrics(
            equity_curve, trade_log, initial_capital=100000
        )
        required = [
            "total_return",
            "annual_return",
            "max_drawdown",
            "sharpe_ratio",
            "volatility",
            "win_rate",
            "total_trades",
            "max_drawdown_duration",
        ]
        for key in required:
            assert key in metrics, f"Missing metric: {key}"

    def test_total_return_positive(self, equity_curve, trade_log):
        metrics = calculate_backtest_metrics(
            equity_curve, trade_log, initial_capital=100000
        )
        assert metrics["total_return"] > 0

    def test_win_rate_calculation(self, equity_curve, trade_log):
        metrics = calculate_backtest_metrics(
            equity_curve, trade_log, initial_capital=100000
        )
        assert 0 <= metrics["win_rate"] <= 1
