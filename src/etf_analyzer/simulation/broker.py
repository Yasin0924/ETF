"""Simulated trade execution for off-exchange ETF. DD-2: No slippage. DD-3: Pending settlement."""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import List
from etf_analyzer.core.calendar import settle_date
from etf_analyzer.simulation.fees import FeeCalculator


class TradeType(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"


@dataclass
class PendingOrder:
    code: str
    trade_type: TradeType
    trade_date: date
    nav: float
    status: OrderStatus = OrderStatus.PENDING
    amount: float = 0.0
    purchase_fee: float = 0.0
    shares: float = 0.0
    confirm_date: date = None
    gross_amount: float = 0.0
    redemption_fee: float = 0.0
    net_amount: float = 0.0
    settle_date: date = None
    holding_days: int = 0


class SimBroker:
    def __init__(self, fee_calculator: FeeCalculator):
        self._fee_calc = fee_calculator
        self._pending_orders: List[PendingOrder] = []

    def submit_buy(
        self,
        code: str,
        amount: float,
        nav: float,
        trade_date: date,
        is_dca: bool = False,
    ) -> PendingOrder:
        fee = self._fee_calc.purchase_fee(amount, is_dca=is_dca)
        net_amount = amount - fee
        shares = net_amount / nav
        conf_date = settle_date(trade_date, n=1)
        order = PendingOrder(
            code=code,
            trade_type=TradeType.BUY,
            trade_date=trade_date,
            nav=nav,
            amount=amount,
            purchase_fee=fee,
            shares=shares,
            confirm_date=conf_date,
        )
        self._pending_orders.append(order)
        return order

    def submit_sell(
        self, code: str, shares: float, nav: float, trade_date: date, holding_days: int
    ) -> PendingOrder:
        gross = shares * nav
        redemption_fee = self._fee_calc.redemption_fee(gross, holding_days)
        net = gross - redemption_fee
        sett_date = settle_date(trade_date, n=2)
        order = PendingOrder(
            code=code,
            trade_type=TradeType.SELL,
            trade_date=trade_date,
            nav=nav,
            shares=shares,
            gross_amount=gross,
            redemption_fee=redemption_fee,
            net_amount=net,
            settle_date=sett_date,
            holding_days=holding_days,
        )
        self._pending_orders.append(order)
        return order

    def process_settlements(self, current_date: date) -> List[PendingOrder]:
        newly_confirmed = []
        still_pending = []
        for order in self._pending_orders:
            target_date = (
                order.confirm_date
                if order.trade_type == TradeType.BUY
                else order.settle_date
            )
            if target_date is not None and target_date <= current_date:
                order.status = OrderStatus.CONFIRMED
                newly_confirmed.append(order)
            else:
                still_pending.append(order)
        self._pending_orders = still_pending
        return newly_confirmed
