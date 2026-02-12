"""Event-driven backtest engine. DD-2: No slippage. DD-3: Settlement processing."""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List
import pandas as pd
from etf_analyzer.core.logger import get_logger
from etf_analyzer.simulation.broker import SimBroker, TradeType
from etf_analyzer.simulation.fees import FeeCalculator, FeeSchedule
from etf_analyzer.simulation.portfolio import Portfolio
from etf_analyzer.strategy.base import BaseStrategy
from etf_analyzer.strategy.signals import SignalType

logger = get_logger("backtest.engine")


@dataclass
class BacktestConfig:
    initial_capital: float = 100000
    start_date: date = date(2020, 1, 1)
    end_date: date = date(2025, 1, 1)
    fee_schedule: FeeSchedule = field(default_factory=FeeSchedule)
    benchmark_code: str = "000300"
    max_trades_per_day: int = 4


class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self._config = config
        fee_calc = FeeCalculator(config.fee_schedule)
        self._broker = SimBroker(fee_calculator=fee_calc)

    def run(self, strategy: BaseStrategy, price_data: pd.DataFrame) -> Dict[str, Any]:
        portfolio = Portfolio(initial_cash=self._config.initial_capital)
        equity_curve = []
        trade_log = []
        etf_codes = [c for c in price_data.columns if c != "日期"]
        for _, row in price_data.iterrows():
            current_date = pd.to_datetime(row["日期"]).date()
            prices: Dict[str, float] = {
                code: float(row[code]) for code in etf_codes if pd.notna(row[code])
            }
            confirmed_orders = self._broker.process_settlements(current_date)
            for order in confirmed_orders:
                if order.trade_type == TradeType.BUY:
                    portfolio.add_lot(
                        code=order.code,
                        shares=order.shares,
                        cost_per_share=order.amount / order.shares,
                        buy_date=order.confirm_date,
                    )
                elif order.trade_type == TradeType.SELL:
                    portfolio.cash += order.net_amount
            trade_count_today = 0
            current_value = portfolio.total_value(prices)
            running_max = (
                max(float(e["total_value"]) for e in equity_curve)
                if equity_curve
                else current_value
            )
            portfolio_drawdown = (
                (current_value - running_max) / running_max if running_max > 0 else 0.0
            )
            valuation_percentiles = {}
            for code in etf_codes:
                series = pd.to_numeric(price_data[code], errors="coerce").dropna()
                if len(series) < 2:
                    continue
                current_price = prices.get(code)
                if current_price is None:
                    continue
                valuation_percentiles[code] = float((series < current_price).mean())
            market_data = {
                "prices": prices,
                "daily_returns": {},
                "trade_count_today": trade_count_today,
                "portfolio_drawdown": portfolio_drawdown,
                "valuation_percentiles": valuation_percentiles,
            }
            signals = strategy.generate_signals(market_data, portfolio, current_date)
            for signal in signals:
                if trade_count_today >= self._config.max_trades_per_day:
                    break
                executed = self._execute_signal(
                    signal, portfolio, prices, current_date, trade_log
                )
                if executed:
                    trade_count_today += 1
            total_value = portfolio.total_value(prices)
            equity_curve.append(
                {
                    "date": current_date,
                    "total_value": total_value,
                    "cash": portfolio.cash,
                }
            )
        final_value = (
            equity_curve[-1]["total_value"]
            if equity_curve
            else self._config.initial_capital
        )
        return {
            "equity_curve": equity_curve,
            "trade_log": trade_log,
            "final_value": final_value,
            "portfolio": portfolio,
        }

    def _execute_signal(
        self,
        signal,
        portfolio: Portfolio,
        prices: Dict[str, float],
        current_date: date,
        trade_log: List[dict],
    ) -> bool:
        code = signal.code
        if signal.signal_type in (SignalType.BUY, SignalType.ADD):
            amount = signal.target_amount
            if amount > portfolio.cash:
                amount = portfolio.cash
            if amount <= 0 or code not in prices:
                return False
            order = self._broker.submit_buy(
                code=code, amount=amount, nav=prices[code], trade_date=current_date
            )
            portfolio.cash -= amount
            trade_log.append(
                {
                    "date": current_date,
                    "code": code,
                    "type": "buy",
                    "amount": amount,
                    "shares": order.shares,
                    "nav": prices[code],
                    "fee": order.purchase_fee,
                    "reason": signal.reason,
                }
            )
            return True
        elif signal.signal_type in (
            SignalType.SELL,
            SignalType.TAKE_PROFIT,
            SignalType.STOP_LOSS,
        ):
            pos = portfolio.get_position(code)
            if pos is None or code not in prices:
                return False
            shares_to_sell = min(signal.target_amount, pos.total_shares)
            if shares_to_sell <= 0:
                return False
            consumed_lots = portfolio.reduce_position_fifo(code, shares_to_sell)
            total_net = 0.0
            total_fee = 0.0
            for lot in consumed_lots:
                holding_days = lot.holding_days(current_date)
                order = self._broker.submit_sell(
                    code=code,
                    shares=lot.shares,
                    nav=prices[code],
                    trade_date=current_date,
                    holding_days=holding_days,
                )
                total_net += order.net_amount
                total_fee += order.redemption_fee
            trade_log.append(
                {
                    "date": current_date,
                    "code": code,
                    "type": signal.signal_type.value,
                    "shares": shares_to_sell,
                    "nav": prices[code],
                    "net_amount": total_net,
                    "fee": total_fee,
                    "reason": signal.reason,
                }
            )
            return True
        return False
