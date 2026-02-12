"""Risk metric formulas. Uses fallback implementations (empyrical not installed)."""

from typing import Tuple
import numpy as np
import pandas as pd

HAS_EMPYRICAL = False


def max_drawdown(nav: pd.Series) -> float:
    if len(nav) <= 1:
        return 0.0
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    return float(drawdown.min())


def max_drawdown_duration(nav: pd.Series) -> int:
    cummax = nav.cummax()
    in_drawdown = nav < cummax
    if not in_drawdown.any():
        return 0
    max_dur = 0
    current_dur = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return max_dur


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    ann_ret = (1 + returns.mean()) ** periods_per_year - 1
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    if ann_vol == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / ann_vol)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    ann_ret = (1 + returns.mean()) ** periods_per_year - 1
    dd = downside_risk(returns, periods_per_year=periods_per_year)
    if dd == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / dd)


def calmar_ratio(returns: pd.Series, nav: pd.Series) -> float:
    ann_ret = (1 + returns.mean()) ** 252 - 1
    mdd = abs(max_drawdown(nav))
    if mdd == 0:
        return float("inf")
    return float(ann_ret / mdd)


def alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> Tuple[float, float]:
    aligned = pd.DataFrame({"r": returns, "b": benchmark_returns}).dropna()
    cov_matrix = aligned.cov()
    beta = cov_matrix.loc["r", "b"] / cov_matrix.loc["b", "b"]
    ann_r = (1 + aligned["r"].mean()) ** 252 - 1
    ann_b = (1 + aligned["b"].mean()) ** 252 - 1
    alpha = (ann_r - risk_free_rate) - beta * (ann_b - risk_free_rate)
    return float(alpha), float(beta)


def downside_risk(returns: pd.Series, periods_per_year: int = 252) -> float:
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return 0.0
    return float(negative_returns.std() * np.sqrt(periods_per_year))


def return_drawdown_ratio(annual_return: float, max_dd: float) -> float:
    if max_dd == 0:
        return float("inf")
    return annual_return / abs(max_dd)
