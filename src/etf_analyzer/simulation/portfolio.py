"""FIFO lot-based portfolio and position tracking. DD-3 applied."""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional


@dataclass
class Lot:
    shares: float
    cost_per_share: float
    buy_date: date

    def market_value(self, price: float) -> float:
        return self.shares * price

    def cost_basis(self) -> float:
        return self.shares * self.cost_per_share

    def holding_days(self, as_of: date) -> int:
        return (as_of - self.buy_date).days


class Position:
    def __init__(self, code: str):
        self.code = code
        self.lots: List[Lot] = []

    @property
    def total_shares(self) -> float:
        return sum(lot.shares for lot in self.lots)

    @property
    def total_cost(self) -> float:
        return sum(lot.cost_basis() for lot in self.lots)

    @property
    def avg_cost(self) -> float:
        total_shares = self.total_shares
        if total_shares == 0:
            return 0.0
        return self.total_cost / total_shares

    def market_value(self, price: float) -> float:
        return self.total_shares * price

    def unrealized_pnl(self, price: float) -> float:
        return self.market_value(price) - self.total_cost

    def unrealized_pnl_pct(self, price: float) -> float:
        if self.total_cost == 0:
            return 0.0
        return self.unrealized_pnl(price) / self.total_cost

    def add_lot(self, shares: float, cost_per_share: float, buy_date: date) -> None:
        self.lots.append(
            Lot(shares=shares, cost_per_share=cost_per_share, buy_date=buy_date)
        )

    def reduce_fifo(self, shares_to_sell: float) -> List[Lot]:
        if shares_to_sell > self.total_shares + 1e-6:
            raise ValueError(
                f"Cannot sell {shares_to_sell} shares; only hold {self.total_shares}"
            )
        consumed = []
        remaining_to_sell = shares_to_sell
        while remaining_to_sell > 1e-6 and self.lots:
            lot = self.lots[0]
            if lot.shares <= remaining_to_sell + 1e-6:
                consumed.append(
                    Lot(
                        shares=lot.shares,
                        cost_per_share=lot.cost_per_share,
                        buy_date=lot.buy_date,
                    )
                )
                remaining_to_sell -= lot.shares
                self.lots.pop(0)
            else:
                consumed.append(
                    Lot(
                        shares=remaining_to_sell,
                        cost_per_share=lot.cost_per_share,
                        buy_date=lot.buy_date,
                    )
                )
                lot.shares -= remaining_to_sell
                remaining_to_sell = 0
        return consumed


class Portfolio:
    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self._initial_cash = initial_cash
        self._positions: Dict[str, Position] = {}

    def add_lot(
        self, code: str, shares: float, cost_per_share: float, buy_date: date
    ) -> None:
        if code not in self._positions:
            self._positions[code] = Position(code=code)
        self._positions[code].add_lot(shares, cost_per_share, buy_date)

    def reduce_position_fifo(self, code: str, shares: float) -> List[Lot]:
        if code not in self._positions:
            raise KeyError(f"No position for {code}")
        consumed = self._positions[code].reduce_fifo(shares)
        if self._positions[code].total_shares < 1e-6:
            del self._positions[code]
        return consumed

    def get_position(self, code: str) -> Optional[Position]:
        return self._positions.get(code)

    @property
    def positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    def total_value(self, prices: Dict[str, float]) -> float:
        pos_value = sum(
            pos.market_value(prices.get(code, pos.avg_cost))
            for code, pos in self._positions.items()
        )
        return self.cash + pos_value

    def position_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        total = self.total_value(prices)
        if total == 0:
            return {}
        return {
            code: pos.market_value(prices.get(code, pos.avg_cost)) / total
            for code, pos in self._positions.items()
        }

    def cash_ratio(self, prices: Dict[str, float]) -> float:
        total = self.total_value(prices)
        if total == 0:
            return 1.0
        return self.cash / total
