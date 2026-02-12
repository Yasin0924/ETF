"""Tests for 3-tier fund screener."""

import pandas as pd
import pytest
from etf_analyzer.selection.screener import FundScreener, ScreeningCriteria


@pytest.fixture
def sample_fund_universe():
    return pd.DataFrame(
        {
            "code": ["510300", "510500", "159915", "512010", "999999"],
            "name": ["沪深300ETF", "中证500ETF", "创业板ETF", "医药ETF", "TinyFund"],
            "scale": [100e8, 50e8, 30e8, 20e8, 2e8],
            "tracking_error": [0.003, 0.005, 0.008, 0.006, 0.015],
            "total_fee_rate": [0.005, 0.005, 0.006, 0.007, 0.008],
            "years_since_inception": [5, 4, 3, 2, 0.5],
            "pe_percentile": [0.15, 0.25, 0.45, 0.10, 0.80],
            "ma60_uptrend": [True, True, False, True, False],
            "annual_volatility": [0.15, 0.18, 0.25, 0.19, 0.30],
            "category": [
                "broad_market",
                "broad_market",
                "broad_market",
                "sector",
                "sector",
            ],
        }
    )


class TestFundScreener:
    def test_initial_screening(self, sample_fund_universe):
        screener = FundScreener()
        result = screener.initial_screen(sample_fund_universe)
        assert "999999" not in result["code"].values
        assert len(result) >= 3

    def test_secondary_screening(self, sample_fund_universe):
        screener = FundScreener()
        initial = screener.initial_screen(sample_fund_universe)
        result = screener.secondary_screen(initial)
        codes = result["code"].tolist()
        assert "510300" in codes
        assert "159915" not in codes

    def test_custom_criteria(self, sample_fund_universe):
        criteria = ScreeningCriteria(min_scale=20e8, max_tracking_error=0.01)
        screener = FundScreener(criteria=criteria)
        result = screener.initial_screen(sample_fund_universe)
        assert all(result["scale"] >= 20e8)
