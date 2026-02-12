"""Shared test fixtures for ETF analyzer tests."""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def sample_nav_series() -> pd.Series:
    """Sample NAV series for 252 trading days (1 year)."""
    dates = pd.bdate_range(start="2024-01-02", periods=252, freq="B")
    np.random.seed(42)
    # Simulate ~8% annual return with ~15% volatility
    daily_returns = np.random.normal(0.08 / 252, 0.15 / np.sqrt(252), 252)
    nav = 1.0 * np.cumprod(1 + daily_returns)
    return pd.Series(nav, index=dates, name="nav")


@pytest.fixture
def sample_daily_returns(sample_nav_series) -> pd.Series:
    """Daily return series derived from NAV."""
    return sample_nav_series.pct_change().dropna()


@pytest.fixture
def sample_benchmark_returns() -> pd.Series:
    """Benchmark (e.g. CSI300) daily returns for 251 trading days."""
    dates = pd.bdate_range(start="2024-01-03", periods=251, freq="B")
    np.random.seed(99)
    returns = np.random.normal(0.06 / 252, 0.18 / np.sqrt(252), 251)
    return pd.Series(returns, index=dates, name="benchmark")


@pytest.fixture
def sample_etf_df() -> pd.DataFrame:
    """Sample ETF daily OHLCV DataFrame (mimics akshare output)."""
    dates = pd.bdate_range(start="2024-01-02", periods=60, freq="B")
    np.random.seed(42)
    close = 1.0 + np.cumsum(np.random.normal(0, 0.01, 60))
    return pd.DataFrame(
        {
            "日期": dates,
            "开盘": close * (1 + np.random.uniform(-0.005, 0.005, 60)),
            "收盘": close,
            "最高": close * (1 + np.abs(np.random.normal(0, 0.005, 60))),
            "最低": close * (1 - np.abs(np.random.normal(0, 0.005, 60))),
            "成交量": np.random.randint(100000, 1000000, 60),
            "成交额": np.random.uniform(1e7, 1e8, 60),
            "涨跌幅": np.random.normal(0, 1, 60),
        }
    )


@pytest.fixture
def sample_pe_history() -> pd.Series:
    """Sample PE ratio history (5 years, ~1260 trading days)."""
    dates = pd.bdate_range(start="2019-01-02", periods=1260, freq="B")
    np.random.seed(42)
    pe = 12 + np.cumsum(np.random.normal(0, 0.1, 1260))
    pe = np.clip(pe, 5, 40)
    return pd.Series(pe, index=dates, name="pe")


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for store tests."""
    etf_dir = tmp_path / "etf"
    index_dir = tmp_path / "index"
    cache_dir = tmp_path / "cache"
    etf_dir.mkdir()
    index_dir.mkdir()
    cache_dir.mkdir()
    return tmp_path
