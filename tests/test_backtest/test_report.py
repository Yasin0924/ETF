"""Tests for backtest visualization and report generation."""

import pandas as pd
import pytest
from pathlib import Path
from etf_analyzer.backtest.visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_position_allocation,
    plot_trade_signals,
    fig_to_base64,
)
from etf_analyzer.backtest.report import generate_html_report


@pytest.fixture
def equity_data():
    dates = pd.bdate_range("2024-01-02", periods=100)
    portfolio = [100000 * (1 + 0.0003) ** i for i in range(100)]
    benchmark = [100000 * (1 + 0.0002) ** i for i in range(100)]
    return pd.DataFrame({"date": dates, "portfolio": portfolio, "benchmark": benchmark})


@pytest.fixture
def weights_data():
    dates = pd.bdate_range("2024-01-02", periods=100)
    w1 = [0.3 + 0.001 * i for i in range(100)]
    w2 = [0.2 - 0.0005 * i for i in range(100)]
    return dates, pd.DataFrame({"510300": w1, "159915": w2})


@pytest.fixture
def trade_log_data():
    from datetime import date

    return [
        {"date": date(2024, 1, 5), "type": "buy", "nav": 1.05, "reason": "PE low"},
        {"date": date(2024, 2, 10), "type": "add", "nav": 1.08, "reason": "rebalance"},
        {
            "date": date(2024, 3, 15),
            "type": "sell",
            "nav": 1.15,
            "reason": "take profit",
        },
        {
            "date": date(2024, 4, 20),
            "type": "stop_loss",
            "nav": 0.95,
            "reason": "drawdown",
        },
        {"date": date(2024, 5, 25), "type": "buy", "nav": 0.90, "reason": "dip buy"},
    ]


class TestVisualization:
    def test_equity_curve_returns_figure(self, equity_data):
        fig = plot_equity_curve(
            dates=equity_data["date"],
            portfolio_values=equity_data["portfolio"],
            benchmark_values=equity_data["benchmark"],
        )
        assert fig is not None

    def test_drawdown_returns_figure(self, equity_data):
        fig = plot_drawdown(
            dates=equity_data["date"], portfolio_values=equity_data["portfolio"]
        )
        assert fig is not None

    def test_fig_to_base64_returns_string(self, equity_data):
        fig = plot_equity_curve(
            dates=equity_data["date"], portfolio_values=equity_data["portfolio"]
        )
        b64 = fig_to_base64(fig)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_position_allocation_returns_figure(self, weights_data):
        dates, weights = weights_data
        fig = plot_position_allocation(dates=dates, weights=weights)
        assert fig is not None

    def test_trade_signals_returns_figure(self, trade_log_data):
        from datetime import date

        dates = pd.Series(pd.bdate_range("2024-01-02", periods=100))
        prices = pd.Series([1.0 + 0.002 * i for i in range(100)])
        fig = plot_trade_signals(dates=dates, prices=prices, trade_log=trade_log_data)
        assert fig is not None

    def test_position_allocation_to_base64(self, weights_data):
        dates, weights = weights_data
        fig = plot_position_allocation(dates=dates, weights=weights)
        b64 = fig_to_base64(fig)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_trade_signals_to_base64(self, trade_log_data):
        dates = pd.Series(pd.bdate_range("2024-01-02", periods=100))
        prices = pd.Series([1.0 + 0.002 * i for i in range(100)])
        fig = plot_trade_signals(dates=dates, prices=prices, trade_log=trade_log_data)
        b64 = fig_to_base64(fig)
        assert isinstance(b64, str)
        assert len(b64) > 100


class TestHtmlReport:
    def test_generate_report_creates_file(self, equity_data, tmp_path):
        output_path = tmp_path / "report.html"
        metrics = {
            "total_return": 0.08,
            "annual_return": 0.08,
            "max_drawdown": -0.05,
            "sharpe_ratio": 1.2,
            "volatility": 0.12,
            "win_rate": 0.6,
            "total_trades": 10,
            "initial_capital": 100000,
            "final_value": 108000,
        }
        generate_html_report(
            metrics=metrics, equity_curve=equity_data, output_path=str(output_path)
        )
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "108000" in content or "108,000" in content
        assert "Portfolio" in content
