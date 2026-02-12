from datetime import date

import pytest

from etf_analyzer.formulas.returns import dca_return
from etf_analyzer.simulation.dca import DCASimulator
from etf_analyzer.simulation.fees import FeeCalculator, FeeSchedule


def _price_series(nav_by_date: dict[date, float]) -> dict[date, float]:
    return nav_by_date


class TestDCASimulator:
    def test_fixed_amount_dca(self):
        fee_calc = FeeCalculator(FeeSchedule(purchase_rate=0.0, dca_discount=1.0))
        simulator = DCASimulator(
            fee_calculator=fee_calc,
            dca_amount=1000.0,
            dca_mode="fixed_amount",
            dca_dates_or_interval=14,
        )
        prices = _price_series(
            {
                date(2024, 1, 1): 1.0,
                date(2024, 1, 15): 1.1,
                date(2024, 1, 29): 0.9,
            }
        )

        result = simulator.simulate(
            price_series=prices,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        expected_shares = 1000 / 1.0 + 1000 / 1.1 + 1000 / 0.9
        assert result.total_invested == pytest.approx(3000.0, abs=1e-9)
        assert result.total_shares == pytest.approx(expected_shares, abs=1e-6)
        assert len(result.purchase_records) == 3

    def test_fixed_shares_dca(self):
        fee_calc = FeeCalculator(FeeSchedule(purchase_rate=0.0, dca_discount=1.0))
        simulator = DCASimulator(
            fee_calculator=fee_calc,
            dca_amount=500.0,
            dca_mode="fixed_shares",
            dca_dates_or_interval=[
                date(2024, 1, 2),
                date(2024, 1, 16),
                date(2024, 1, 30),
            ],
        )
        prices = _price_series(
            {
                date(2024, 1, 2): 1.0,
                date(2024, 1, 16): 2.0,
                date(2024, 1, 30): 4.0,
            }
        )

        result = simulator.simulate(
            price_series=prices,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert result.total_shares == pytest.approx(1500.0, abs=1e-9)
        assert result.total_invested == pytest.approx(3500.0, abs=1e-9)
        assert result.avg_cost == pytest.approx(3500.0 / 1500.0, abs=1e-9)

    def test_holiday_deferral(self):
        fee_calc = FeeCalculator(FeeSchedule(purchase_rate=0.0, dca_discount=1.0))
        simulator = DCASimulator(
            fee_calculator=fee_calc,
            dca_amount=1000.0,
            dca_mode="fixed_amount",
            dca_dates_or_interval=[date(2024, 1, 6)],
        )
        prices = _price_series({date(2024, 1, 8): 1.0})

        result = simulator.simulate(
            price_series=prices,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
        )

        assert len(result.purchase_records) == 1
        assert result.purchase_records[0]["scheduled_date"] == date(2024, 1, 6)
        assert result.purchase_records[0]["trade_date"] == date(2024, 1, 8)

    def test_dca_with_fees(self):
        fee_calc = FeeCalculator(FeeSchedule(purchase_rate=0.015, dca_discount=0.1))
        simulator = DCASimulator(
            fee_calculator=fee_calc,
            dca_amount=1000.0,
            dca_mode="fixed_amount",
            dca_dates_or_interval=[date(2024, 1, 2)],
        )
        prices = _price_series({date(2024, 1, 2): 1.0})

        result = simulator.simulate(
            price_series=prices,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
        )

        assert result.total_invested == pytest.approx(1000.0, abs=1e-9)
        assert result.total_shares == pytest.approx(998.5, abs=1e-6)
        assert result.purchase_records[0]["fee"] == pytest.approx(1.5, abs=1e-9)

    def test_insufficient_funds(self):
        fee_calc = FeeCalculator(FeeSchedule(purchase_rate=0.0, dca_discount=1.0))
        simulator = DCASimulator(
            fee_calculator=fee_calc,
            dca_amount=1000.0,
            dca_mode="fixed_amount",
            dca_dates_or_interval=[date(2024, 1, 2), date(2024, 1, 9)],
        )
        simulator.available_cash = 1500.0
        prices = _price_series({date(2024, 1, 2): 1.0, date(2024, 1, 9): 1.0})

        with pytest.raises(ValueError, match="insufficient funds"):
            simulator.simulate(
                price_series=prices,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 10),
            )

    def test_dca_return_calculation(self):
        fee_calc = FeeCalculator(FeeSchedule(purchase_rate=0.0, dca_discount=1.0))
        simulator = DCASimulator(
            fee_calculator=fee_calc,
            dca_amount=1000.0,
            dca_mode="fixed_amount",
            dca_dates_or_interval=[
                date(2024, 1, 2),
                date(2024, 1, 9),
                date(2024, 1, 16),
            ],
        )
        prices = _price_series(
            {
                date(2024, 1, 2): 1.0,
                date(2024, 1, 9): 0.8,
                date(2024, 1, 16): 1.2,
                date(2024, 1, 31): 1.1,
            }
        )

        result = simulator.simulate(
            price_series=prices,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        expected = dca_return(
            [1.0, 0.8, 1.2], amount_per_purchase=1000.0, final_nav=1.1
        )
        assert result.total_return == pytest.approx(expected, abs=1e-9)
