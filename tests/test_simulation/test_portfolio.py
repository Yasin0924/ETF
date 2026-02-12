"""Tests for FIFO lot-based portfolio. DD-3 applied."""

from datetime import date
import pytest
from etf_analyzer.simulation.portfolio import Portfolio, Lot


class TestPortfolio:
    def test_initial_cash(self):
        p = Portfolio(initial_cash=100000)
        assert p.cash == 100000
        assert p.total_value(prices={}) == 100000

    def test_add_lot(self):
        p = Portfolio(initial_cash=100000)
        p.add_lot("510300", shares=1000, cost_per_share=1.5, buy_date=date(2024, 1, 2))
        p.cash -= 1500
        pos = p.get_position("510300")
        assert pos is not None
        assert pos.total_shares == 1000
        assert pos.avg_cost == 1.5

    def test_multiple_lots_tracked_separately(self):
        p = Portfolio(initial_cash=100000)
        p.add_lot("510300", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        p.add_lot("510300", shares=500, cost_per_share=2.0, buy_date=date(2024, 2, 1))
        pos = p.get_position("510300")
        assert pos.total_shares == 1500
        assert len(pos.lots) == 2
        assert pos.avg_cost == pytest.approx(1.333, abs=0.01)

    def test_reduce_position_fifo(self):
        p = Portfolio(initial_cash=100000)
        p.add_lot("510300", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        p.add_lot("510300", shares=500, cost_per_share=2.0, buy_date=date(2024, 2, 1))
        consumed_lots = p.reduce_position_fifo("510300", shares=800)
        assert len(consumed_lots) == 1
        assert consumed_lots[0].shares == 800
        assert consumed_lots[0].buy_date == date(2024, 1, 2)
        pos = p.get_position("510300")
        assert pos.total_shares == 700
        assert len(pos.lots) == 2
        assert pos.lots[0].shares == 200

    def test_reduce_fifo_spans_multiple_lots(self):
        p = Portfolio(initial_cash=100000)
        p.add_lot("510300", shares=300, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        p.add_lot("510300", shares=500, cost_per_share=2.0, buy_date=date(2024, 2, 1))
        consumed_lots = p.reduce_position_fifo("510300", shares=500)
        assert len(consumed_lots) == 2
        assert consumed_lots[0].shares == 300
        assert consumed_lots[1].shares == 200
        pos = p.get_position("510300")
        assert pos.total_shares == 300
        assert len(pos.lots) == 1

    def test_reduce_all_removes_position(self):
        p = Portfolio(initial_cash=100000)
        p.add_lot("510300", shares=1000, cost_per_share=1.5, buy_date=date(2024, 1, 2))
        p.reduce_position_fifo("510300", shares=1000)
        assert p.get_position("510300") is None

    def test_total_value_with_positions(self):
        p = Portfolio(initial_cash=50000)
        p.add_lot("510300", shares=1000, cost_per_share=1.5, buy_date=date(2024, 1, 2))
        prices = {"510300": 2.0}
        assert p.total_value(prices) == 50000 + 1000 * 2.0

    def test_position_weights(self):
        p = Portfolio(initial_cash=0)
        p.add_lot("510300", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        p.add_lot("510500", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        weights = p.position_weights(prices={"510300": 2.0, "510500": 3.0})
        assert weights["510300"] == pytest.approx(0.4, abs=0.01)
        assert weights["510500"] == pytest.approx(0.6, abs=0.01)

    def test_lot_holding_days(self):
        p = Portfolio(initial_cash=100000)
        p.add_lot("510300", shares=1000, cost_per_share=1.5, buy_date=date(2024, 1, 2))
        p.add_lot("510300", shares=500, cost_per_share=1.6, buy_date=date(2024, 3, 1))
        pos = p.get_position("510300")
        assert pos.lots[0].holding_days(date(2024, 4, 1)) == 90
        assert pos.lots[1].holding_days(date(2024, 4, 1)) == 31
