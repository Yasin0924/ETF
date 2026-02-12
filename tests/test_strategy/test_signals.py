"""Tests for trading signal types."""

from datetime import date
import pytest
from etf_analyzer.strategy.signals import Signal, SignalType


class TestSignalType:
    def test_all_enum_values_exist(self):
        assert SignalType.BUY.value == "buy"
        assert SignalType.SELL.value == "sell"
        assert SignalType.ADD.value == "add"
        assert SignalType.HOLD.value == "hold"
        assert SignalType.REBALANCE.value == "rebalance"
        assert SignalType.TAKE_PROFIT.value == "take_profit"
        assert SignalType.STOP_LOSS.value == "stop_loss"

    def test_enum_count(self):
        assert len(SignalType) == 7

    def test_enum_from_value(self):
        assert SignalType("buy") is SignalType.BUY
        assert SignalType("stop_loss") is SignalType.STOP_LOSS

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            SignalType("invalid")


class TestSignal:
    def test_creation_required_fields(self):
        sig = Signal(signal_type=SignalType.BUY, code="510300", reason="PE low")
        assert sig.signal_type is SignalType.BUY
        assert sig.code == "510300"
        assert sig.reason == "PE low"

    def test_default_values(self):
        sig = Signal(signal_type=SignalType.HOLD, code="510300", reason="hold")
        assert sig.strength == 1.0
        assert sig.target_amount == 0.0
        assert sig.target_weight == 0.0
        assert sig.date is None

    def test_optional_fields(self):
        d = date(2024, 6, 15)
        sig = Signal(
            signal_type=SignalType.ADD,
            code="159915",
            reason="rebalance add",
            strength=0.8,
            target_amount=5000.0,
            target_weight=0.15,
            date=d,
        )
        assert sig.strength == 0.8
        assert sig.target_amount == 5000.0
        assert sig.target_weight == 0.15
        assert sig.date == d

    def test_equality(self):
        a = Signal(signal_type=SignalType.SELL, code="510300", reason="stop")
        b = Signal(signal_type=SignalType.SELL, code="510300", reason="stop")
        assert a == b

    def test_inequality(self):
        a = Signal(signal_type=SignalType.BUY, code="510300", reason="low PE")
        b = Signal(signal_type=SignalType.SELL, code="510300", reason="low PE")
        assert a != b

    def test_signal_in_list(self):
        signals = [
            Signal(signal_type=SignalType.BUY, code="510300", reason="buy"),
            Signal(signal_type=SignalType.SELL, code="159915", reason="sell"),
            Signal(signal_type=SignalType.HOLD, code="518880", reason="hold"),
        ]
        assert len(signals) == 3
        assert all(isinstance(s, Signal) for s in signals)
        buy_signals = [s for s in signals if s.signal_type is SignalType.BUY]
        assert len(buy_signals) == 1

    def test_sell_type_signals(self):
        sell_types = {SignalType.SELL, SignalType.TAKE_PROFIT, SignalType.STOP_LOSS}
        for st in sell_types:
            sig = Signal(signal_type=st, code="510300", reason="test")
            assert sig.signal_type in sell_types
