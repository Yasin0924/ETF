"""Tests for signal types and base strategy."""

import pytest
from etf_analyzer.strategy.signals import Signal, SignalType
from etf_analyzer.strategy.base import BaseStrategy


class TestSignal:
    def test_create_buy_signal(self):
        sig = Signal(
            signal_type=SignalType.BUY,
            code="510300",
            reason="PE below 20% percentile",
            strength=0.8,
        )
        assert sig.signal_type == SignalType.BUY
        assert sig.code == "510300"
        assert sig.strength == 0.8

    def test_signal_types_exist(self):
        assert SignalType.BUY
        assert SignalType.SELL
        assert SignalType.HOLD
        assert SignalType.REBALANCE
        assert SignalType.ADD
        assert SignalType.TAKE_PROFIT
        assert SignalType.STOP_LOSS


class TestBaseStrategy:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            BaseStrategy(name="test")

    def test_subclass_must_implement_generate_signals(self):
        class IncompleteStrategy(BaseStrategy):
            pass

        with pytest.raises(TypeError):
            IncompleteStrategy(name="test")

    def test_subclass_works(self):
        class DummyStrategy(BaseStrategy):
            def generate_signals(self, market_data, portfolio, current_date):
                return []

        s = DummyStrategy(name="dummy")
        assert s.name == "dummy"
        assert s.generate_signals({}, None, None) == []
