from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite
from typing import Iterable

from etf_analyzer.core.calendar import is_trading_day, next_trading_day
from etf_analyzer.simulation.fees import FeeCalculator
from etf_analyzer.simulation.portfolio import Portfolio


@dataclass
class DCAResult:
    total_invested: float
    total_shares: float
    avg_cost: float
    current_value: float
    total_return: float
    purchase_records: list[dict]


class DCASimulator:
    def __init__(
        self,
        fee_calculator: FeeCalculator,
        dca_amount: float,
        dca_mode: str,
        dca_dates_or_interval,
    ):
        if dca_mode not in {"fixed_amount", "fixed_shares"}:
            raise ValueError("dca_mode must be 'fixed_amount' or 'fixed_shares'")
        if dca_amount <= 0:
            raise ValueError("dca_amount must be positive")

        self.fee_calculator = fee_calculator
        self.dca_amount = dca_amount
        self.dca_mode = dca_mode
        self.dca_dates_or_interval = dca_dates_or_interval
        self.available_cash = float("inf")

    def _scheduled_dates(self, start_date: date, end_date: date) -> list[date]:
        if isinstance(self.dca_dates_or_interval, int):
            interval_days = self.dca_dates_or_interval
            if interval_days <= 0:
                raise ValueError("dca_dates_or_interval interval must be positive")
            dates = []
            current = start_date
            while current <= end_date:
                dates.append(current)
                current += timedelta(days=interval_days)
            return dates

        if not isinstance(self.dca_dates_or_interval, Iterable):
            raise ValueError("dca_dates_or_interval must be int or iterable of dates")

        selected = [
            d
            for d in self.dca_dates_or_interval
            if isinstance(d, date) and start_date <= d <= end_date
        ]
        return sorted(selected)

    def _resolve_trade_date(
        self, scheduled_date: date, end_date: date, nav_lookup: dict[date, float]
    ) -> date | None:
        candidate = scheduled_date
        if not is_trading_day(candidate):
            candidate = next_trading_day(candidate)

        while candidate <= end_date:
            if candidate in nav_lookup:
                return candidate
            candidate = next_trading_day(candidate)
        return None

    def _fixed_shares_amount(self, nav: float) -> tuple[float, float, float]:
        fee_rate = self.fee_calculator.purchase_fee(1.0, is_dca=True)
        if fee_rate >= 1:
            raise ValueError("effective purchase fee rate must be less than 1")
        net_amount = self.dca_amount * nav
        gross_amount = net_amount / (1 - fee_rate)
        fee = self.fee_calculator.purchase_fee(gross_amount, is_dca=True)
        return gross_amount, fee, self.dca_amount

    def simulate(
        self,
        price_series: dict[date, float],
        start_date: date,
        end_date: date,
    ) -> DCAResult:
        if start_date > end_date:
            raise ValueError("start_date must be <= end_date")
        if not price_series:
            raise ValueError("price_series cannot be empty")

        nav_lookup = dict(price_series)
        portfolio = Portfolio(
            initial_cash=self.available_cash if isfinite(self.available_cash) else 0.0
        )
        cash_remaining = self.available_cash
        purchase_records: list[dict] = []
        total_invested = 0.0

        for scheduled_date in self._scheduled_dates(start_date, end_date):
            trade_date = self._resolve_trade_date(scheduled_date, end_date, nav_lookup)
            if trade_date is None:
                continue

            nav = nav_lookup[trade_date]
            if nav <= 0:
                raise ValueError("NAV must be positive")

            if self.dca_mode == "fixed_amount":
                gross_amount = self.dca_amount
                fee = self.fee_calculator.purchase_fee(gross_amount, is_dca=True)
                shares = (gross_amount - fee) / nav
            else:
                gross_amount, fee, shares = self._fixed_shares_amount(nav)

            if gross_amount > cash_remaining + 1e-9:
                raise ValueError("insufficient funds")

            cash_remaining -= gross_amount
            total_invested += gross_amount

            portfolio.add_lot(
                code="DCA",
                shares=shares,
                cost_per_share=gross_amount / shares,
                buy_date=trade_date,
            )

            purchase_records.append(
                {
                    "scheduled_date": scheduled_date,
                    "trade_date": trade_date,
                    "nav": nav,
                    "amount": gross_amount,
                    "fee": fee,
                    "shares": shares,
                    "industry_dispersion": {},
                    "style_dispersion": {},
                }
            )

        final_nav_date = max(d for d in nav_lookup if start_date <= d <= end_date)
        final_nav = nav_lookup[final_nav_date]
        position = portfolio.get_position("DCA")
        total_shares = position.total_shares if position is not None else 0.0
        avg_cost = position.avg_cost if position is not None else 0.0
        current_value = total_shares * final_nav
        total_return = (
            (current_value - total_invested) / total_invested
            if total_invested > 0
            else 0.0
        )

        return DCAResult(
            total_invested=total_invested,
            total_shares=total_shares,
            avg_cost=avg_cost,
            current_value=current_value,
            total_return=total_return,
            purchase_records=purchase_records,
        )
