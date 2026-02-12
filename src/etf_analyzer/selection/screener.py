"""Multi-tier fund screening."""

from dataclasses import dataclass
import pandas as pd
from etf_analyzer.core.logger import get_logger

logger = get_logger("selection.screener")


@dataclass
class ScreeningCriteria:
    min_scale: float = 5e8
    max_tracking_error: float = 0.01
    max_total_fee: float = 0.006
    min_years: float = 1.0
    max_pe_percentile: float = 0.30
    require_ma_uptrend: bool = True
    max_volatility: float = 0.20


class FundScreener:
    def __init__(self, criteria: ScreeningCriteria = None):
        self._criteria = criteria or ScreeningCriteria()

    def initial_screen(self, universe: pd.DataFrame) -> pd.DataFrame:
        c = self._criteria
        mask = (
            (universe["scale"] >= c.min_scale)
            & (universe["tracking_error"] <= c.max_tracking_error)
            & (universe["total_fee_rate"] <= c.max_total_fee)
            & (universe["years_since_inception"] >= c.min_years)
        )
        result = universe[mask].reset_index(drop=True)
        logger.info(f"Initial screening: {len(universe)} -> {len(result)} funds")
        return result

    def secondary_screen(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self._criteria
        mask = (df["pe_percentile"] <= c.max_pe_percentile) & (
            df["annual_volatility"] <= c.max_volatility
        )
        if c.require_ma_uptrend:
            mask = mask & (df["ma60_uptrend"] == True)
        result = df[mask].reset_index(drop=True)
        logger.info(f"Secondary screening: {len(df)} -> {len(result)} funds")
        return result

    def screen(self, universe: pd.DataFrame) -> pd.DataFrame:
        initial = self.initial_screen(universe)
        return self.secondary_screen(initial)
