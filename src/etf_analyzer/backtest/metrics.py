"""Backtest performance metrics aggregation."""

from typing import Any, Dict, List
import pandas as pd
from etf_analyzer.formulas.risk import (
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    annualized_volatility,
)
from etf_analyzer.formulas.returns import annualized_return


def calculate_backtest_metrics(
    equity_curve: List[dict],
    trade_log: List[dict],
    initial_capital: float,
    risk_free_rate: float = 0.02,
) -> Dict[str, Any]:
    if not equity_curve:
        return {}
    values = pd.Series(
        [e["total_value"] for e in equity_curve],
        index=pd.DatetimeIndex([e["date"] for e in equity_curve]),
    )
    daily_returns = values.pct_change().dropna()
    final_value = values.iloc[-1]
    total_ret = (final_value - initial_capital) / initial_capital
    holding_days = (values.index[-1] - values.index[0]).days
    if holding_days <= 0:
        holding_days = 1
    ann_ret = annualized_return(total_ret, holding_days)
    mdd = max_drawdown(values)
    mdd_dur = max_drawdown_duration(values)
    vol = annualized_volatility(daily_returns) if len(daily_returns) > 1 else 0.0
    sr = sharpe_ratio(daily_returns, risk_free_rate) if len(daily_returns) > 1 else 0.0
    sell_trades = [
        t for t in trade_log if t.get("type") in ("sell", "take_profit", "stop_loss")
    ]
    buy_trades = [t for t in trade_log if t.get("type") == "buy"]
    total_trades = len(buy_trades) + len(sell_trades)
    wins = 0
    for t in sell_trades:
        net = t.get("net_amount", 0)
        amount = t.get("amount", 0)
        if net > amount or net > 0:
            wins += 1
    win_rate = wins / len(sell_trades) if sell_trades else 0.0
    return {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return": total_ret,
        "annual_return": ann_ret,
        "max_drawdown": mdd,
        "max_drawdown_duration": mdd_dur,
        "sharpe_ratio": sr,
        "volatility": vol,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "holding_days": holding_days,
    }
