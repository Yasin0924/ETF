"""Fund scoring and ranking. Weighted: valuation(40%) + fees(30%) + tracking_error(30%)."""

import numpy as np
import pandas as pd
from etf_analyzer.core.logger import get_logger

logger = get_logger("selection.scorer")


class FundScorer:
    def __init__(
        self,
        valuation_weight: float = 0.4,
        fee_weight: float = 0.3,
        tracking_weight: float = 0.3,
    ):
        self._val_w = valuation_weight
        self._fee_w = fee_weight
        self._te_w = tracking_weight

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["val_score"] = 1.0 - result["pe_percentile"]
        fee_min = result["total_fee_rate"].min()
        fee_max = result["total_fee_rate"].max()
        if fee_max > fee_min:
            result["fee_score"] = 1.0 - (result["total_fee_rate"] - fee_min) / (
                fee_max - fee_min
            )
        else:
            result["fee_score"] = 1.0
        te_min = result["tracking_error"].min()
        te_max = result["tracking_error"].max()
        if te_max > te_min:
            result["te_score"] = 1.0 - (result["tracking_error"] - te_min) / (
                te_max - te_min
            )
        else:
            result["te_score"] = 1.0
        result["total_score"] = (
            result["val_score"] * self._val_w
            + result["fee_score"] * self._fee_w
            + result["te_score"] * self._te_w
        )
        return result

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        scored = self.score(df)
        return scored.sort_values("total_score", ascending=False).reset_index(drop=True)

    def top_n_per_category(self, df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        ranked = self.rank(df)
        if "category" not in ranked.columns:
            return ranked.head(n)
        return ranked.groupby("category").head(n).reset_index(drop=True)
