"""Semi-monthly rebalance strategy implementation."""

from datetime import date
from typing import Any, Dict, List
from etf_analyzer.strategy.base import BaseStrategy
from etf_analyzer.strategy.signals import Signal, SignalType
from etf_analyzer.simulation.portfolio import Portfolio
from etf_analyzer.core.logger import get_logger

logger = get_logger("strategy.semi_monthly")


class SemiMonthlyStrategy(BaseStrategy):
    def __init__(self, params: dict):
        super().__init__(name="semi_monthly_rebalance", params=params)
        self._rebalance_days = params.get("rebalance_day", [1, 16])
        self._deviation_single = params.get("deviation_single", 0.05)
        self._target_weights = params.get("target_weights", {})
        self._tp_config = params.get("take_profit", {})
        self._sl_config = params.get("stop_loss", {})
        self._buy_config = params.get("buy_signal", {})
        self._min_cash_ratio = params.get("min_cash_ratio", 0.05)
        self._daily_trade_limit = int(params.get("max_trades_per_day", 999))

    def generate_signals(
        self, market_data: Dict[str, Any], portfolio: Portfolio, current_date: date
    ) -> List[Signal]:
        signals = []
        prices = market_data.get("prices", {})
        daily_returns = market_data.get("daily_returns", {})
        valuation_percentiles = market_data.get("valuation_percentiles", {})
        portfolio_drawdown = float(market_data.get("portfolio_drawdown", 0.0))
        trade_count_today = int(market_data.get("trade_count_today", 0))

        force_reduce = self._check_portfolio_drawdown_force_reduce(
            portfolio, prices, portfolio_drawdown
        )
        signals.extend(force_reduce)

        pause_add = self._is_pause_add(portfolio_drawdown)
        if not pause_add and trade_count_today < self._daily_trade_limit:
            signals.extend(
                self._check_buy_add(
                    portfolio=portfolio,
                    prices=prices,
                    daily_returns=daily_returns,
                    valuation_percentiles=valuation_percentiles,
                )
            )

        signals.extend(self._check_stop_loss(portfolio, prices, daily_returns))
        signals.extend(self._check_take_profit(portfolio, prices))
        if current_date.day in self._rebalance_days:
            signals.extend(self._check_rebalance(portfolio, prices))
        return signals

    def _resolve_category(self, code: str) -> str:
        if code == "518880":
            return "gold"
        return "broad_market"

    def _is_pause_add(self, portfolio_drawdown: float) -> bool:
        cfg = self._sl_config.get("portfolio_drawdown", {})
        pause_threshold = cfg.get("pause_add_threshold", -0.10)
        return portfolio_drawdown <= pause_threshold

    def _check_portfolio_drawdown_force_reduce(
        self,
        portfolio: Portfolio,
        prices: Dict[str, float],
        portfolio_drawdown: float,
    ) -> List[Signal]:
        cfg = self._sl_config.get("portfolio_drawdown", {})
        threshold = cfg.get("force_reduce_threshold", -0.15)
        reduce_ratio = cfg.get("force_reduce_ratio", 0.2)
        if portfolio_drawdown > threshold:
            return []
        signals = []
        for code, pos in portfolio.positions.items():
            if code not in prices or pos.total_shares <= 0:
                continue
            signals.append(
                Signal(
                    signal_type=SignalType.STOP_LOSS,
                    code=code,
                    reason=(
                        f"组合回撤 {portfolio_drawdown:.1%} 触发强制减仓阈值 {threshold:.1%}"
                    ),
                    target_amount=pos.total_shares * reduce_ratio,
                )
            )
        return signals

    def _check_buy_add(
        self,
        portfolio: Portfolio,
        prices: Dict[str, float],
        daily_returns: Dict[str, float],
        valuation_percentiles: Dict[str, float],
    ) -> List[Signal]:
        signals = []
        if not prices:
            return signals
        cash_ratio = portfolio.cash_ratio(prices)
        if cash_ratio < self._min_cash_ratio:
            return signals

        for code, price in prices.items():
            if price <= 0:
                continue
            category = self._resolve_category(code)
            cfg = self._buy_config.get(category, {})
            if not cfg:
                continue
            trigger_drop = float(cfg.get("daily_drop_trigger", -0.03))
            daily_ret = float(daily_returns.get(code, 0.0))
            val_pct = valuation_percentiles.get(code)
            pe_threshold = cfg.get("pe_percentile_threshold")
            value_trigger = (
                pe_threshold is not None
                and val_pct is not None
                and val_pct <= pe_threshold
            )
            drop_trigger = daily_ret <= trigger_drop
            if not (value_trigger or drop_trigger):
                continue

            has_pos = portfolio.get_position(code) is not None
            base_amount = max(portfolio.cash * 0.1, 1000)
            amount = min(base_amount, portfolio.cash)
            if amount <= 0:
                continue

            signal_type = SignalType.ADD if has_pos else SignalType.BUY
            reason = (
                f"估值/跌幅触发：估值分位={val_pct}, 当日涨跌={daily_ret:.2%}"
                if value_trigger
                else f"单日下跌触发：当日涨跌={daily_ret:.2%}"
            )
            signals.append(
                Signal(
                    signal_type=signal_type,
                    code=code,
                    reason=reason,
                    target_amount=amount,
                )
            )
        return signals

    def _check_rebalance(
        self, portfolio: Portfolio, prices: Dict[str, float]
    ) -> List[Signal]:
        signals = []
        if not self._target_weights or not prices:
            return signals
        current_weights = portfolio.position_weights(prices)
        for code, target_w in self._target_weights.items():
            current_w = current_weights.get(code, 0.0)
            deviation = current_w - target_w
            if abs(deviation) > self._deviation_single:
                signals.append(
                    Signal(
                        signal_type=SignalType.REBALANCE,
                        code=code,
                        reason=f"仓位偏离 {deviation:+.2%} 超过阈值 ±{self._deviation_single:.0%}",
                        target_weight=target_w,
                    )
                )
        return signals

    def _check_take_profit(
        self, portfolio: Portfolio, prices: Dict[str, float]
    ) -> List[Signal]:
        signals = []
        for code, pos in portfolio.positions.items():
            if code not in prices:
                continue
            pnl_pct = pos.unrealized_pnl_pct(prices[code])
            tier2 = self._tp_config.get("tier2", {})
            tier1 = self._tp_config.get("tier1", {})
            if tier2 and pnl_pct >= tier2.get("return_threshold", 0.30):
                signals.append(
                    Signal(
                        signal_type=SignalType.TAKE_PROFIT,
                        code=code,
                        reason=f"收益率 {pnl_pct:.1%} 达到二档止盈阈值 {tier2['return_threshold']:.0%}",
                        target_amount=pos.total_shares
                        * tier2.get("reduce_ratio", 0.30),
                    )
                )
            elif tier1 and pnl_pct >= tier1.get("return_threshold", 0.15):
                signals.append(
                    Signal(
                        signal_type=SignalType.TAKE_PROFIT,
                        code=code,
                        reason=f"收益率 {pnl_pct:.1%} 达到一档止盈阈值 {tier1['return_threshold']:.0%}",
                        target_amount=pos.total_shares
                        * tier1.get("reduce_ratio", 0.20),
                    )
                )
        return signals

    def _check_stop_loss(
        self,
        portfolio: Portfolio,
        prices: Dict[str, float],
        daily_returns: Dict[str, float],
    ) -> List[Signal]:
        signals = []
        max_dd_threshold = self._sl_config.get("single_max_drawdown", -0.20)
        for code, pos in portfolio.positions.items():
            if code not in prices:
                continue
            pnl_pct = pos.unrealized_pnl_pct(prices[code])
            if pnl_pct <= max_dd_threshold:
                signals.append(
                    Signal(
                        signal_type=SignalType.STOP_LOSS,
                        code=code,
                        reason=f"亏损 {pnl_pct:.1%} 超过最大回撤阈值 {max_dd_threshold:.0%}",
                        target_amount=pos.total_shares,
                    )
                )
        return signals
