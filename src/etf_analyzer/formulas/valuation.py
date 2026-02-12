"""Valuation indicator formulas. PB unavailable from akshare (DD-1)."""

import pandas as pd


def percentile_rank(current_value: float, history: pd.Series) -> float:
    if len(history) == 0:
        raise ValueError("History cannot be empty")
    return float((history < current_value).sum() / len(history))


def valuation_zone(
    percentile: float,
    low_threshold: float = 0.30,
    high_threshold: float = 0.70,
) -> str:
    if percentile <= low_threshold:
        return "undervalued"
    elif percentile > high_threshold:
        return "overvalued"
    return "normal"


def dividend_yield(annual_dividend: float, price: float) -> float:
    if price <= 0:
        raise ValueError("price must be positive")
    return annual_dividend / price
