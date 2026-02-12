"""End-to-end integration test: strategy -> backtest -> metrics -> report."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_analyzer.backtest.engine import BacktestEngine, BacktestConfig
from etf_analyzer.backtest.metrics import calculate_backtest_metrics
from etf_analyzer.backtest.report import generate_html_report
from etf_analyzer.simulation.fees import FeeSchedule
from etf_analyzer.strategy.base import BaseStrategy
from etf_analyzer.strategy.signals import Signal, SignalType


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy-and-hold for integration testing."""

    def generate_signals(self, market_data, portfolio, current_date):
        prices = market_data.get("prices", {})
        for code in prices:
            if not portfolio.get_position(code):
                target = portfolio.cash * 0.45  # Buy ~45% per ETF
                if target > 100:
                    return [
                        Signal(
                            signal_type=SignalType.BUY,
                            code=code,
                            reason="Initial allocation",
                            target_amount=target,
                        )
                    ]
        return []


@pytest.fixture
def synthetic_price_data():
    """Generate 1 year of synthetic price data for 2 ETFs."""
    dates = pd.bdate_range("2024-01-02", periods=252)
    np.random.seed(42)
    etf1 = 1.0 * np.cumprod(1 + np.random.normal(0.0003, 0.01, 252))
    etf2 = 2.0 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 252))
    return pd.DataFrame(
        {
            "日期": dates,
            "510300": etf1,
            "510500": etf2,
        }
    )


class TestEndToEnd:
    def test_full_pipeline(self, synthetic_price_data, tmp_path):
        # 1. Configure and run backtest
        config = BacktestConfig(
            initial_capital=100000,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 12, 31),
            fee_schedule=FeeSchedule(),
        )
        engine = BacktestEngine(config=config)
        strategy = BuyAndHoldStrategy(name="buy_hold")
        result = engine.run(strategy=strategy, price_data=synthetic_price_data)

        # 2. Verify backtest output
        assert result["final_value"] > 0
        assert len(result["equity_curve"]) == 252
        assert len(result["trade_log"]) >= 1

        # 3. Calculate metrics
        metrics = calculate_backtest_metrics(
            equity_curve=result["equity_curve"],
            trade_log=result["trade_log"],
            initial_capital=100000,
        )
        assert "annual_return" in metrics
        assert "max_drawdown" in metrics
        assert "sharpe_ratio" in metrics

        # 4. Generate report
        equity_df = pd.DataFrame(result["equity_curve"])
        equity_df = equity_df.rename(columns={"total_value": "portfolio"})
        report_path = tmp_path / "backtest_report.html"
        generate_html_report(
            metrics=metrics,
            equity_curve=equity_df,
            output_path=str(report_path),
        )
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Portfolio" in content
        assert len(content) > 500

    def test_pipeline_with_zero_trades(self, tmp_path):
        """Strategy that never trades should still produce valid results."""

        class NoOpStrategy(BaseStrategy):
            def generate_signals(self, market_data, portfolio, current_date):
                return []

        dates = pd.bdate_range("2024-01-02", periods=20)
        price_data = pd.DataFrame(
            {
                "日期": dates,
                "510300": [1.0 + i * 0.01 for i in range(20)],
            }
        )
        config = BacktestConfig(initial_capital=100000, fee_schedule=FeeSchedule())
        engine = BacktestEngine(config=config)
        result = engine.run(strategy=NoOpStrategy(name="noop"), price_data=price_data)
        assert result["final_value"] == 100000  # All cash
        assert len(result["trade_log"]) == 0

        metrics = calculate_backtest_metrics(
            result["equity_curve"], result["trade_log"], 100000
        )
        assert metrics["total_return"] == pytest.approx(0.0, abs=1e-9)
