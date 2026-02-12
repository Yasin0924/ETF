"""Tests for simulated trade execution broker."""

from datetime import date
import pytest
from etf_analyzer.simulation.broker import (
    SimBroker,
    PendingOrder,
    OrderStatus,
    TradeType,
)
from etf_analyzer.simulation.fees import FeeCalculator, FeeSchedule


class TestSimBroker:
    @pytest.fixture
    def broker(self):
        schedule = FeeSchedule(purchase_rate=0.015, discount=0.1)
        fee_calc = FeeCalculator(schedule)
        return SimBroker(fee_calculator=fee_calc)

    def test_buy_order_returns_pending(self, broker):
        order = broker.submit_buy(
            code="510300", amount=10000.0, nav=1.5, trade_date=date(2024, 1, 2)
        )
        assert order.status == OrderStatus.PENDING
        assert order.trade_type == TradeType.BUY
        assert order.confirm_date == date(2024, 1, 3)

    def test_sell_order_returns_pending(self, broker):
        order = broker.submit_sell(
            code="510300",
            shares=1000.0,
            nav=1.5,
            trade_date=date(2024, 1, 2),
            holding_days=100,
        )
        assert order.status == OrderStatus.PENDING
        assert order.trade_type == TradeType.SELL
        assert order.settle_date == date(2024, 1, 4)

    def test_buy_at_exact_nav_no_slippage(self, broker):
        order = broker.submit_buy(
            code="510300", amount=10000.0, nav=1.0, trade_date=date(2024, 1, 2)
        )
        assert order.shares == pytest.approx(9985.0, abs=0.01)

    def test_sell_at_exact_nav_no_slippage(self, broker):
        order = broker.submit_sell(
            code="510300",
            shares=1000.0,
            nav=1.5,
            trade_date=date(2024, 1, 2),
            holding_days=100,
        )
        assert order.gross_amount == pytest.approx(1500.0, abs=0.01)
        assert order.net_amount == pytest.approx(1496.25, abs=0.01)

    def test_process_settlements_confirms_buy(self, broker):
        order = broker.submit_buy(
            code="510300", amount=10000.0, nav=1.0, trade_date=date(2024, 1, 2)
        )
        assert order.status == OrderStatus.PENDING
        confirmed = broker.process_settlements(date(2024, 1, 2))
        assert len(confirmed) == 0
        confirmed = broker.process_settlements(date(2024, 1, 3))
        assert len(confirmed) == 1
        assert confirmed[0].status == OrderStatus.CONFIRMED

    def test_process_settlements_settles_sell(self, broker):
        order = broker.submit_sell(
            code="510300",
            shares=1000.0,
            nav=1.5,
            trade_date=date(2024, 1, 2),
            holding_days=100,
        )
        settled = broker.process_settlements(date(2024, 1, 3))
        assert len(settled) == 0
        settled = broker.process_settlements(date(2024, 1, 4))
        assert len(settled) == 1
        assert settled[0].status == OrderStatus.CONFIRMED
