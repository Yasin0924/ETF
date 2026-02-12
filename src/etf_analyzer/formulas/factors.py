"""Strategy factor formulas."""

import numpy as np
import pandas as pd


def momentum_factor(prices: pd.Series, lookback: int = 20) -> float:
    if len(prices) < lookback + 1:
        raise ValueError(
            f"Insufficient data: need {lookback + 1} points, got {len(prices)}"
        )
    return float(
        (prices.iloc[-1] - prices.iloc[-lookback - 1]) / prices.iloc[-lookback - 1]
    )


def quality_factor(
    tracking_error: float,
    fund_scale: float,
    management_fee: float,
    te_weight: float = 0.4,
    scale_weight: float = 0.3,
    fee_weight: float = 0.3,
) -> float:
    te_score = np.clip(1.0 - (tracking_error - 0.005) / 0.015, 0, 1)
    scale_score = np.clip((fund_scale - 1e8) / (10e8 - 1e8), 0, 1)
    fee_score = np.clip(1.0 - (management_fee - 0.002) / 0.008, 0, 1)
    return float(
        te_score * te_weight + scale_score * scale_weight + fee_score * fee_weight
    )
