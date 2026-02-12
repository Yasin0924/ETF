"""Tests for fee calculation."""

import pytest
from etf_analyzer.simulation.fees import FeeCalculator, FeeSchedule


class TestFeeCalculator:
    def test_purchase_fee(self):
        schedule = FeeSchedule(purchase_rate=0.015, discount=0.1)
        calc = FeeCalculator(schedule)
        fee = calc.purchase_fee(amount=10000)
        assert fee == pytest.approx(15.0, abs=0.01)

    def test_redemption_fee_short_hold(self):
        schedule = FeeSchedule(
            redemption_tiers=[
                (7, 0.015),
                (30, 0.005),
                (365, 0.0025),
                (float("inf"), 0.0),
            ]
        )
        calc = FeeCalculator(schedule)
        fee = calc.redemption_fee(amount=10000, holding_days=3)
        assert fee == pytest.approx(150.0, abs=0.01)

    def test_redemption_fee_long_hold(self):
        schedule = FeeSchedule(
            redemption_tiers=[
                (7, 0.015),
                (30, 0.005),
                (365, 0.0025),
                (float("inf"), 0.0),
            ]
        )
        calc = FeeCalculator(schedule)
        fee = calc.redemption_fee(amount=10000, holding_days=400)
        assert fee == pytest.approx(0.0, abs=0.01)

    def test_daily_management_fee_informational(self):
        schedule = FeeSchedule(management_rate=0.005, custody_rate=0.001)
        calc = FeeCalculator(schedule)
        daily_fee = calc.daily_accrued_fee(nav_total=1000000)
        expected = (0.005 + 0.001) / 365 * 1000000
        assert daily_fee == pytest.approx(expected, abs=0.1)

    def test_dca_purchase_discount(self):
        schedule = FeeSchedule(purchase_rate=0.015, dca_discount=0.1)
        calc = FeeCalculator(schedule)
        fee = calc.purchase_fee(amount=1000, is_dca=True)
        assert fee == pytest.approx(1.5, abs=0.01)
