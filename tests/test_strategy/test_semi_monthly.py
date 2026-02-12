"""Tests for semi-monthly rebalance strategy."""

from datetime import date
import pytest
from etf_analyzer.strategy.semi_monthly import SemiMonthlyStrategy
from etf_analyzer.strategy.signals import SignalType
from etf_analyzer.simulation.portfolio import Portfolio


@pytest.fixture
def strategy():
    params = {
        "rebalance_day": [1, 16],
        "deviation_single": 0.05,
        "deviation_portfolio": 0.03,
        "buy_signal": {
            "broad_market": {
                "pe_percentile_threshold": 0.20,
                "daily_drop_trigger": -0.03,
            }
        },
        "take_profit": {
            "tier1": {"return_threshold": 0.15, "reduce_ratio": 0.20},
            "tier2": {"return_threshold": 0.30, "reduce_ratio": 0.30},
        },
        "stop_loss": {
            "single_max_drawdown": -0.20,
            "ma_break": {
                "ma_period": 20,
                "daily_drop": -0.05,
                "confirm_days": 3,
                "reduce_ratio": 0.50,
            },
            "portfolio_drawdown": {
                "pause_add_threshold": -0.10,
                "force_reduce_threshold": -0.15,
                "force_reduce_ratio": 0.20,
            },
        },
        "target_weights": {"510300": 0.40, "510500": 0.30, "518880": 0.30},
    }
    return SemiMonthlyStrategy(params=params)


class TestSemiMonthlyStrategy:
    def test_buy_signal_when_no_position_and_valuation_low(self, strategy):
        portfolio = Portfolio(initial_cash=100000)
        market_data = {
            "prices": {"510300": 1.0},
            "daily_returns": {"510300": -0.04},
            "valuation_percentiles": {"510300": 0.18},
        }
        signals = strategy.generate_signals(market_data, portfolio, date(2024, 1, 10))
        assert any(
            s.signal_type == SignalType.BUY and s.code == "510300" for s in signals
        )

    def test_pause_add_under_portfolio_drawdown(self, strategy):
        portfolio = Portfolio(initial_cash=100000)
        portfolio.add_lot(
            "510300", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2)
        )
        market_data = {
            "prices": {"510300": 0.9},
            "daily_returns": {"510300": -0.03},
            "valuation_percentiles": {"510300": 0.15},
            "portfolio_drawdown": -0.12,
        }
        signals = strategy.generate_signals(market_data, portfolio, date(2024, 2, 10))
        assert not any(
            s.signal_type in {SignalType.BUY, SignalType.ADD} for s in signals
        )

    def test_force_reduce_under_deep_drawdown(self, strategy):
        portfolio = Portfolio(initial_cash=100000)
        portfolio.add_lot(
            "510300", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2)
        )
        market_data = {
            "prices": {"510300": 0.95},
            "daily_returns": {"510300": -0.01},
            "portfolio_drawdown": -0.16,
        }
        signals = strategy.generate_signals(market_data, portfolio, date(2024, 2, 10))
        assert any(
            s.signal_type == SignalType.STOP_LOSS and "组合回撤" in s.reason
            for s in signals
        )

    def test_rebalance_signal_on_rebalance_day(self, strategy):
        portfolio = Portfolio(initial_cash=100000)
        portfolio.add_lot(
            "510300", shares=5000, cost_per_share=1.0, buy_date=date(2024, 1, 2)
        )
        portfolio.add_lot(
            "510500", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2)
        )
        portfolio.add_lot(
            "518880", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2)
        )
        portfolio.cash = 93000
        market_data = {
            "prices": {"510300": 1.2, "510500": 1.0, "518880": 1.0},
            "daily_returns": {"510300": 0.01, "510500": -0.005, "518880": 0.0},
        }
        signals = strategy.generate_signals(market_data, portfolio, date(2024, 1, 16))
        signal_types = [s.signal_type for s in signals]
        assert SignalType.REBALANCE in signal_types

    def test_no_rebalance_on_non_rebalance_day(self, strategy):
        portfolio = Portfolio(initial_cash=100000)
        market_data = {"prices": {}, "daily_returns": {}}
        signals = strategy.generate_signals(market_data, portfolio, date(2024, 1, 10))
        rebalance_signals = [
            s for s in signals if s.signal_type == SignalType.REBALANCE
        ]
        assert len(rebalance_signals) == 0

    def test_take_profit_signal(self, strategy):
        portfolio = Portfolio(initial_cash=100000)
        portfolio.add_lot(
            "510300", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2)
        )
        portfolio.cash = 99000
        market_data = {"prices": {"510300": 1.20}, "daily_returns": {"510300": 0.01}}
        signals = strategy.generate_signals(market_data, portfolio, date(2024, 3, 1))
        tp_signals = [s for s in signals if s.signal_type == SignalType.TAKE_PROFIT]
        assert len(tp_signals) > 0

    def test_stop_loss_on_large_drawdown(self, strategy):
        portfolio = Portfolio(initial_cash=100000)
        portfolio.add_lot(
            "510300", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2)
        )
        portfolio.cash = 99000
        market_data = {"prices": {"510300": 0.75}, "daily_returns": {"510300": -0.06}}
        signals = strategy.generate_signals(market_data, portfolio, date(2024, 3, 1))
        sl_signals = [s for s in signals if s.signal_type == SignalType.STOP_LOSS]
        assert len(sl_signals) > 0
