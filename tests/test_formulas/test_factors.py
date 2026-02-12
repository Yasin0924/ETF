"""Tests for strategy factor formulas."""

import numpy as np
import pandas as pd
import pytest
from etf_analyzer.formulas.factors import (
    momentum_factor,
    quality_factor,
)


class TestMomentumFactor:
    def test_positive_momentum(self):
        prices = pd.Series(range(1, 253), dtype=float)
        mom = momentum_factor(prices, lookback=20)
        assert mom > 0

    def test_negative_momentum(self):
        prices = pd.Series(range(252, 0, -1), dtype=float)
        mom = momentum_factor(prices, lookback=20)
        assert mom < 0

    def test_insufficient_data_raises(self):
        prices = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="Insufficient data"):
            momentum_factor(prices, lookback=20)


class TestQualityFactor:
    def test_low_tracking_error_high_quality(self):
        score = quality_factor(
            tracking_error=0.005,
            fund_scale=10e8,
            management_fee=0.003,
        )
        assert score > 0.7

    def test_high_tracking_error_low_quality(self):
        score = quality_factor(
            tracking_error=0.02,
            fund_scale=2e8,
            management_fee=0.01,
        )
        assert score < 0.5

    def test_output_range(self):
        score = quality_factor(
            tracking_error=0.01,
            fund_scale=5e8,
            management_fee=0.005,
        )
        assert 0 <= score <= 1
