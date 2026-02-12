"""Tests for backtest engine."""

from datetime import date
import pandas as pd
import pytest
from etf_analyzer.backtest.engine import BacktestEngine, BacktestConfig
from etf_analyzer.strategy.base import BaseStrategy
from etf_analyzer.strategy.signals import Signal, SignalType
from etf_analyzer.simulation.fees import FeeSchedule


class MockStrategy(BaseStrategy):
    def generate_signals(self, market_data, portfolio, current_date):
        if not portfolio.get_position("510300"):
            return [
                Signal(
                    signal_type=SignalType.BUY,
                    code="510300",
                    reason="Initial buy",
                    target_amount=50000,
                )
            ]
        return []


@pytest.fixture
def price_data():
    dates = pd.bdate_range("2024-01-02", periods=60)
    prices = [1.0 + i * 0.01 for i in range(60)]
    return pd.DataFrame({"日期": dates, "510300": prices})


class TestBacktestEngine:
    def test_basic_backtest_runs(self, price_data):
        config = BacktestConfig(
            initial_capital=100000,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 22),
            fee_schedule=FeeSchedule(),
        )
        engine = BacktestEngine(config=config)
        strategy = MockStrategy(name="mock")
        result = engine.run(strategy=strategy, price_data=price_data)
        assert result is not None
        assert "equity_curve" in result
        assert "trade_log" in result
        assert "final_value" in result
        assert result["final_value"] > 0

    def test_equity_curve_length(self, price_data):
        config = BacktestConfig(
            initial_capital=100000,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 22),
            fee_schedule=FeeSchedule(),
        )
        engine = BacktestEngine(config=config)
        strategy = MockStrategy(name="mock")
        result = engine.run(strategy=strategy, price_data=price_data)
        assert len(result["equity_curve"]) == len(price_data)

    def test_trade_log_records_buy(self, price_data):
        config = BacktestConfig(
            initial_capital=100000,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 22),
            fee_schedule=FeeSchedule(),
        )
        engine = BacktestEngine(config=config)
        strategy = MockStrategy(name="mock")
        result = engine.run(strategy=strategy, price_data=price_data)
        assert len(result["trade_log"]) >= 1
        assert result["trade_log"][0]["type"] == "buy"
