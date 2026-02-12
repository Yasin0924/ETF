"""Technical analysis formulas."""

import numpy as np
import pandas as pd


def moving_average(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window).mean()


def is_ma_uptrend(prices: pd.Series, window: int = 60) -> bool:
    ma = moving_average(prices, window)
    if ma.isna().iloc[-1]:
        return False
    return bool(prices.iloc[-1] > ma.iloc[-1])


def daily_change_pct(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def drawdown_from_peak(nav: pd.Series) -> pd.Series:
    cummax = nav.cummax()
    return (nav - cummax) / cummax


def rolling_volatility(
    returns: pd.Series, window: int = 20, annualize: bool = False
) -> pd.Series:
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol
