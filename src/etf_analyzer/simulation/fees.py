"""Fee calculation for off-exchange ETF linked funds. DD-2: daily_accrued_fee for reporting only."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class FeeSchedule:
    purchase_rate: float = 0.015
    discount: float = 0.1
    dca_discount: float = 0.1
    redemption_tiers: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (7, 0.015),
            (30, 0.005),
            (365, 0.0025),
            (float("inf"), 0.0),
        ]
    )
    management_rate: float = 0.005
    custody_rate: float = 0.001
    sales_service_rate: float = 0.0


class FeeCalculator:
    def __init__(self, schedule: FeeSchedule):
        self._schedule = schedule

    def purchase_fee(self, amount: float, is_dca: bool = False) -> float:
        discount = self._schedule.dca_discount if is_dca else self._schedule.discount
        return amount * self._schedule.purchase_rate * discount

    def redemption_fee(self, amount: float, holding_days: int) -> float:
        for max_days, rate in self._schedule.redemption_tiers:
            if holding_days < max_days:
                return amount * rate
        return 0.0

    def daily_accrued_fee(self, nav_total: float) -> float:
        annual_rate = (
            self._schedule.management_rate
            + self._schedule.custody_rate
            + self._schedule.sales_service_rate
        )
        return nav_total * annual_rate / 365
