"""Tests for return calculation formulas."""

import numpy as np
import pandas as pd
import pytest
from etf_analyzer.formulas.returns import (
    holding_period_return,
    annualized_return,
    weighted_portfolio_return,
    dca_return,
)


class TestHoldingPeriodReturn:
    def test_basic_return(self):
        result = holding_period_return(buy_nav=1.0, sell_nav=1.1)
        assert result == pytest.approx(0.1, abs=1e-9)

    def test_negative_return(self):
        result = holding_period_return(buy_nav=1.0, sell_nav=0.9)
        assert result == pytest.approx(-0.1, abs=1e-9)

    def test_zero_buy_nav_raises(self):
        with pytest.raises(ValueError, match="buy_nav must be positive"):
            holding_period_return(buy_nav=0, sell_nav=1.0)


class TestAnnualizedReturn:
    def test_one_year_holding(self):
        result = annualized_return(total_return=0.1, holding_days=365)
        assert result == pytest.approx(0.1, abs=1e-4)

    def test_two_year_holding(self):
        result = annualized_return(total_return=0.21, holding_days=730)
        assert result == pytest.approx(0.1, abs=1e-2)

    def test_zero_days_raises(self):
        with pytest.raises(ValueError, match="holding_days must be positive"):
            annualized_return(total_return=0.1, holding_days=0)


class TestWeightedPortfolioReturn:
    def test_equal_weights(self):
        returns = [0.1, 0.2, 0.3]
        weights = [1 / 3, 1 / 3, 1 / 3]
        result = weighted_portfolio_return(returns, weights)
        assert result == pytest.approx(0.2, abs=1e-9)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="Weights must sum to 1"):
            weighted_portfolio_return([0.1, 0.2], [0.5, 0.6])


class TestDcaReturn:
    def test_fixed_amount_dca(self):
        nav_at_purchase = [1.0, 0.8, 1.2]
        amount_per_purchase = 1000.0
        final_nav = 1.1
        result = dca_return(nav_at_purchase, amount_per_purchase, final_nav)
        assert result == pytest.approx(0.1306, abs=1e-3)
