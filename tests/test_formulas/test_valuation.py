"""Tests for valuation indicator formulas."""

import numpy as np
import pandas as pd
import pytest
from etf_analyzer.formulas.valuation import (
    percentile_rank,
    valuation_zone,
    dividend_yield,
)


class TestPercentileRank:
    def test_current_at_median(self):
        history = pd.Series(range(1, 101))
        result = percentile_rank(current_value=50, history=history)
        assert result == pytest.approx(0.49, abs=0.02)

    def test_current_at_min(self):
        history = pd.Series(range(1, 101))
        result = percentile_rank(current_value=1, history=history)
        assert result == pytest.approx(0.0, abs=0.02)

    def test_current_at_max(self):
        history = pd.Series(range(1, 101))
        result = percentile_rank(current_value=100, history=history)
        assert result == pytest.approx(0.99, abs=0.02)

    def test_empty_history_raises(self):
        with pytest.raises(ValueError, match="History cannot be empty"):
            percentile_rank(current_value=10, history=pd.Series([], dtype=float))


class TestValuationZone:
    def test_undervalued(self):
        assert valuation_zone(percentile=0.15) == "undervalued"

    def test_normal(self):
        assert valuation_zone(percentile=0.50) == "normal"

    def test_overvalued(self):
        assert valuation_zone(percentile=0.85) == "overvalued"

    def test_boundary_30(self):
        assert valuation_zone(percentile=0.30) == "undervalued"

    def test_boundary_70(self):
        assert valuation_zone(percentile=0.70) == "normal"


class TestDividendYield:
    def test_basic_yield(self):
        result = dividend_yield(annual_dividend=0.5, price=10.0)
        assert result == pytest.approx(0.05, abs=1e-9)

    def test_zero_price_raises(self):
        with pytest.raises(ValueError):
            dividend_yield(annual_dividend=0.5, price=0)
