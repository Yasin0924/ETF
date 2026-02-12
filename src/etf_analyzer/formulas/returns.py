"""Return calculation formulas."""

from typing import List
import numpy as np


def holding_period_return(buy_nav: float, sell_nav: float) -> float:
    if buy_nav <= 0:
        raise ValueError("buy_nav must be positive")
    return (sell_nav - buy_nav) / buy_nav


def annualized_return(total_return: float, holding_days: int) -> float:
    if holding_days <= 0:
        raise ValueError("holding_days must be positive")
    return (1 + total_return) ** (365 / holding_days) - 1


def weighted_portfolio_return(returns: List[float], weights: List[float]) -> float:
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1")
    return float(np.dot(returns, weights))


def dca_return(
    nav_at_purchase: List[float],
    amount_per_purchase: float,
    final_nav: float,
) -> float:
    total_shares = sum(amount_per_purchase / nav for nav in nav_at_purchase)
    total_cost = amount_per_purchase * len(nav_at_purchase)
    final_value = total_shares * final_nav
    return (final_value - total_cost) / total_cost
