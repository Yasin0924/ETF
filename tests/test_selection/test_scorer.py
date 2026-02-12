"""Tests for fund scoring and ranking."""

import pandas as pd
import pytest
from etf_analyzer.selection.scorer import FundScorer


@pytest.fixture
def screened_funds():
    return pd.DataFrame(
        {
            "code": ["510300", "510500", "512010"],
            "pe_percentile": [0.10, 0.20, 0.15],
            "total_fee_rate": [0.004, 0.005, 0.006],
            "tracking_error": [0.003, 0.005, 0.008],
            "category": ["broad_market", "broad_market", "sector"],
        }
    )


class TestFundScorer:
    def test_score_all_funds(self, screened_funds):
        scorer = FundScorer()
        scored = scorer.score(screened_funds)
        assert "total_score" in scored.columns
        assert scored["total_score"].between(0, 1).all()

    def test_ranking_order(self, screened_funds):
        scorer = FundScorer()
        ranked = scorer.rank(screened_funds)
        scores = ranked["total_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_top_n_per_category(self, screened_funds):
        scorer = FundScorer()
        top = scorer.top_n_per_category(screened_funds, n=1)
        categories = top["category"].unique()
        for cat in categories:
            assert len(top[top["category"] == cat]) <= 1

    def test_custom_weights(self, screened_funds):
        scorer = FundScorer(valuation_weight=0.5, fee_weight=0.3, tracking_weight=0.2)
        scored = scorer.score(screened_funds)
        assert "total_score" in scored.columns
