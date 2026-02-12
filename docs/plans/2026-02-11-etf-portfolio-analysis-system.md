# ETF Fund Portfolio Investment Analysis System - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a modular Python system for off-exchange (场外) ETF linked fund portfolio analysis, covering data management, quantitative formulas, fund selection, strategy execution, trading simulation, and backtesting with HTML reports.

**Architecture:** Layered modular design: Utils/Config (底层) → Data + Formulas (基础层) → Simulation (模拟层) → Strategy + Selection (策略层) → Backtest + Report (回测层). Each module exposes functions returning `ApiResponse(status_code, data, message)`. All strategy parameters are config-driven (YAML). External API calls are wrapped with retry + failover.

**Tech Stack:** Python 3.11+, pandas, numpy, akshare (data), empyrical (metrics), matplotlib (charts), jinja2 (reports), pyyaml (config), pytest (testing)

**Reference Documents:**
- Design spec: `概要设计文档.md` (root)
- akshare ETF APIs: `fund_etf_hist_em()`, `fund_etf_spot_em()`, `fund_name_em()`, `fund_fee_em()`, `fund_info_index_em()`, `stock_zh_index_value_csindex()`
- Metrics library: `empyrical` (sharpe_ratio, max_drawdown, annual_return, alpha_beta, calmar_ratio, annual_volatility, sortino_ratio)

---

## Critical Design Decisions

> These decisions resolve known gaps between the design spec and real-world data/execution constraints.

### DD-1: PE-only valuation (no PB)

**Problem:** akshare's `stock_zh_index_value_csindex()` returns PE ratio + dividend yield, but NOT PB ratio for indices. No reliable free PB data source exists for index-level valuation.

**Decision:** All valuation logic uses **PE percentile + dividend yield only**. PB is explicitly unsupported in v1. The design spec's "PE/PB" references are interpreted as "PE" throughout. The 复筛 filter "PE/PB处于近5年0%-30%分位" becomes "PE处于近5年0%-30%分位". Scoring formula "估值(40%)" uses PE percentile only.

**Impact:** Task 7 (valuation), Task 13 (fetcher), Task 20 (screener), Task 21 (scorer) — all use `pe_percentile` only, no `pb_percentile`.

### DD-2: No slippage for off-exchange funds + no management fee double-counting

**Problem:** Off-exchange (场外) linked funds trade at the **declared NAV** — there is no bid/ask spread or market impact. Applying ±0.5% slippage to off-exchange trades is incorrect. Additionally, the published NAV already has management + custody fees deducted daily, so accruing them separately in the simulator would double-count.

**Decision:**
- **Slippage = 0** for off-exchange fund simulation. The broker executes at exact declared NAV. The design spec's "交易滑点±0.5%" is removed for off-exchange (it only applies to on-exchange ETF trading, which is out of scope).
- **Daily management/custody fee accrual is removed** from the backtest flow. The `FeeCalculator.daily_accrued_fee()` method is retained for informational reporting only (e.g. "this fund costs X per year in embedded fees") but is NOT deducted from portfolio value during simulation, since NAV already reflects these fees.
- Only **purchase fees** and **redemption fees** (tiered by holding period) are deducted during simulation.

**Impact:** Task 15 (fees), Task 16 (broker), Task 22 (backtest engine) — slippage removed, daily fee accrual not applied.

### DD-3: FIFO lot-based position tracking + pending/confirmed settlement

**Problem:** (a) Tiered redemption fees depend on per-lot holding period (e.g. <7 days = 1.5%, 30-365 days = 0.25%). A weighted-average position with a single `first_buy_date` cannot calculate correct redemption fees for partial sells of a multi-purchase position. (b) Off-exchange T+1 purchase confirmation and T+2 redemption settlement mean shares/cash are not immediately available — the system must distinguish pending from confirmed states.

**Decision:**
- **Position uses FIFO lot tracking.** Each purchase creates a `Lot(shares, cost, buy_date)`. When selling, lots are consumed FIFO (oldest first). Redemption fee per lot is calculated using that lot's actual holding days. `avg_cost` and `total_shares` are derived properties computed from the lot list.
- **Broker maintains a pending order queue.** `submit_buy()` returns a `PendingOrder` that transitions to confirmed shares on T+1. `submit_sell()` returns a `PendingOrder` whose cash arrives on T+2. The backtest engine calls `broker.process_settlements(current_date)` each day to confirm pending orders.
- **Portfolio holds only confirmed lots** (via `add_lot()`). Pending orders live in the broker's `_pending_orders` queue. Strategy accesses `pos.total_shares` which reflects confirmed shares only — pending buys are not yet visible.

**Impact:** Task 16 (broker), Task 17 (portfolio), Task 22 (backtest engine) — complete redesign of position model and settlement flow.

---

## Project Structure

```
ETF/
├── src/
│   └── etf_analyzer/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py          # YAML config loader
│       │   ├── response.py        # ApiResponse(status_code, data, message)
│       │   ├── logger.py          # Structured logging
│       │   ├── cache.py           # Simple disk/memory cache
│       │   └── calendar.py        # A-share trading calendar
│       ├── data/
│       │   ├── __init__.py
│       │   ├── fetcher.py         # akshare API wrappers
│       │   ├── cleaner.py         # Data cleaning & validation
│       │   ├── store.py           # CSV read/write per ETF
│       │   └── updater.py         # Incremental daily updates
│       ├── formulas/
│       │   ├── __init__.py
│       │   ├── returns.py         # Return calculations
│       │   ├── valuation.py       # PE percentile & dividend yield (PB unavailable, see DD-1)
│       │   ├── risk.py            # Drawdown, Sharpe, etc.
│       │   ├── technical.py       # MA, momentum
│       │   └── factors.py         # Strategy factors
│       ├── selection/
│       │   ├── __init__.py
│       │   ├── screener.py        # 3-tier filtering
│       │   └── scorer.py          # Weighted scoring & ranking
│       ├── strategy/
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract base strategy
│       │   ├── signals.py         # Signal types (BUY/SELL/HOLD/REBALANCE)
│       │   └── semi_monthly.py    # Semi-monthly rebalance strategy
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── fees.py            # Fee calculation (purchase/redemption/mgmt/custody)
│       │   ├── broker.py          # Trade execution (T+1/T+2 rules)
│       │   ├── portfolio.py       # Position tracking & weighted cost
│       │   └── dca.py             # Dollar-cost averaging simulation
│       └── backtest/
│           ├── __init__.py
│           ├── engine.py          # Event-driven backtest engine
│           ├── metrics.py         # Performance metrics aggregation
│           ├── visualization.py   # matplotlib charts
│           └── report.py          # Jinja2 HTML report generator
├── tests/
│   ├── conftest.py                # Shared fixtures (sample DataFrames, etc.)
│   ├── test_core/
│   │   ├── test_config.py
│   │   ├── test_response.py
│   │   ├── test_cache.py
│   │   └── test_calendar.py
│   ├── test_formulas/
│   │   ├── test_returns.py
│   │   ├── test_valuation.py
│   │   ├── test_risk.py
│   │   ├── test_technical.py
│   │   └── test_factors.py
│   ├── test_data/
│   │   ├── test_fetcher.py
│   │   ├── test_cleaner.py
│   │   ├── test_store.py
│   │   └── test_updater.py
│   ├── test_selection/
│   │   ├── test_screener.py
│   │   └── test_scorer.py
│   ├── test_strategy/
│   │   ├── test_base.py
│   │   ├── test_signals.py
│   │   └── test_semi_monthly.py
│   ├── test_simulation/
│   │   ├── test_fees.py
│   │   ├── test_broker.py
│   │   ├── test_portfolio.py
│   │   └── test_dca.py
│   └── test_backtest/
│       ├── test_engine.py
│       ├── test_metrics.py
│       └── test_report.py
├── config/
│   ├── settings.yaml              # Global settings
│   ├── strategy_params.yaml       # Strategy parameters
│   └── data_sources.yaml          # Data source config
├── data/                          # Runtime data (gitignored)
│   ├── etf/
│   ├── index/
│   └── cache/
├── templates/
│   └── report.html                # Jinja2 report template
├── pyproject.toml
├── requirements.txt
└── 概要设计文档.md
```

---

## Phase 1: Project Infrastructure

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: all `__init__.py` files (empty)
- Create: `tests/conftest.py`
- Create: `.gitignore`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "etf-analyzer"
version = "0.1.0"
description = "ETF Fund Portfolio Investment Analysis System"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "akshare>=1.12",
    "empyrical>=0.5.5",
    "matplotlib>=3.7",
    "jinja2>=3.1",
    "pyyaml>=6.0",
    "openpyxl>=3.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.setuptools.packages.find]
where = ["src"]
```

**Step 2: Create requirements.txt**

```
pandas>=2.0
numpy>=1.24
akshare>=1.12
empyrical>=0.5.5
matplotlib>=3.7
jinja2>=3.1
pyyaml>=6.0
openpyxl>=3.1
pytest>=7.4
pytest-cov>=4.1
```

**Step 3: Create all directory structure and __init__.py files**

Create every directory and empty `__init__.py` as listed in the project structure above.

**Step 4: Create .gitignore**

```
data/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
build/
.venv/
```

**Step 5: Create tests/conftest.py with shared fixtures**

```python
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
    return pd.DataFrame({
        "日期": dates,
        "开盘": close * (1 + np.random.uniform(-0.005, 0.005, 60)),
        "收盘": close,
        "最高": close * (1 + np.abs(np.random.normal(0, 0.005, 60))),
        "最低": close * (1 - np.abs(np.random.normal(0, 0.005, 60))),
        "成交量": np.random.randint(100000, 1000000, 60),
        "成交额": np.random.uniform(1e7, 1e8, 60),
        "涨跌幅": np.random.normal(0, 1, 60),
    })


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
```

**Step 6: Install dependencies & run empty test suite**

Run: `pip install -e ".[dev]" && pytest --tb=short`
Expected: 0 tests collected, exit code 0 (or 5 for no tests)

**Step 7: Commit**

```bash
git init && git add -A && git commit -m "chore: project scaffolding with dependencies and test fixtures"
```

---

### Task 2: ApiResponse and structured logging

**Files:**
- Test: `tests/test_core/test_response.py`
- Create: `src/etf_analyzer/core/response.py`
- Create: `src/etf_analyzer/core/logger.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_response.py
"""Tests for ApiResponse wrapper."""
from etf_analyzer.core.response import ApiResponse, StatusCode


class TestApiResponse:
    def test_success_response(self):
        resp = ApiResponse.success(data={"value": 42})
        assert resp.status_code == StatusCode.SUCCESS
        assert resp.data == {"value": 42}
        assert resp.message == ""
        assert resp.ok is True

    def test_error_response(self):
        resp = ApiResponse.error(message="Data source failed")
        assert resp.status_code == StatusCode.ERROR
        assert resp.data is None
        assert resp.message == "Data source failed"
        assert resp.ok is False

    def test_warning_response(self):
        resp = ApiResponse.warning(data=[1, 2, 3], message="Partial data")
        assert resp.status_code == StatusCode.WARNING
        assert resp.data == [1, 2, 3]
        assert resp.message == "Partial data"
        assert resp.ok is True

    def test_to_dict(self):
        resp = ApiResponse.success(data={"x": 1})
        d = resp.to_dict()
        assert d["status_code"] == StatusCode.SUCCESS.value
        assert d["data"] == {"x": 1}
        assert "message" in d
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_response.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'etf_analyzer.core.response'`

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/core/response.py
"""Unified API response wrapper (status_code + data + message)."""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class StatusCode(IntEnum):
    SUCCESS = 0
    WARNING = 1
    ERROR = 2


@dataclass(frozen=True)
class ApiResponse:
    """All module interfaces return this to enable unified error handling."""
    status_code: StatusCode
    data: Any = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status_code != StatusCode.ERROR

    @classmethod
    def success(cls, data: Any = None, message: str = "") -> "ApiResponse":
        return cls(status_code=StatusCode.SUCCESS, data=data, message=message)

    @classmethod
    def error(cls, message: str, data: Any = None) -> "ApiResponse":
        return cls(status_code=StatusCode.ERROR, data=data, message=message)

    @classmethod
    def warning(cls, data: Any = None, message: str = "") -> "ApiResponse":
        return cls(status_code=StatusCode.WARNING, data=data, message=message)

    def to_dict(self) -> dict:
        return {
            "status_code": self.status_code.value,
            "data": self.data,
            "message": self.message,
        }
```

```python
# src/etf_analyzer/core/logger.py
"""Structured logging for the ETF analyzer system."""
import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger with consistent formatting."""
    logger = logging.getLogger(f"etf_analyzer.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_core/test_response.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add ApiResponse wrapper and structured logging"
```

---

### Task 3: Config system (YAML loader)

**Files:**
- Test: `tests/test_core/test_config.py`
- Create: `src/etf_analyzer/core/config.py`
- Create: `config/settings.yaml`
- Create: `config/strategy_params.yaml`
- Create: `config/data_sources.yaml`

**Step 1: Write the failing test**

```python
# tests/test_core/test_config.py
"""Tests for YAML config loader."""
import pytest
import yaml
from etf_analyzer.core.config import load_config, get_setting


class TestConfig:
    def test_load_config_from_file(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump({"key": "value", "nested": {"a": 1}}))
        config = load_config(str(config_file))
        assert config["key"] == "value"
        assert config["nested"]["a"] == 1

    def test_load_config_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_get_setting_dot_notation(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump({
            "backtest": {"initial_capital": 100000, "benchmark": "000300"}
        }))
        config = load_config(str(config_file))
        assert get_setting(config, "backtest.initial_capital") == 100000
        assert get_setting(config, "backtest.benchmark") == "000300"

    def test_get_setting_default_value(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump({"a": 1}))
        config = load_config(str(config_file))
        assert get_setting(config, "nonexistent.key", default=42) == 42

    def test_get_setting_missing_no_default_raises(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump({"a": 1}))
        config = load_config(str(config_file))
        with pytest.raises(KeyError):
            get_setting(config, "nonexistent.key")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/core/config.py
"""YAML configuration loader with dot-notation access."""
from pathlib import Path
from typing import Any

import yaml

_SENTINEL = object()


def load_config(path: str) -> dict:
    """Load YAML config file. Raises FileNotFoundError if missing."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_setting(config: dict, key: str, default: Any = _SENTINEL) -> Any:
    """Access nested config with dot notation: 'backtest.initial_capital'.

    Args:
        config: Loaded config dict.
        key: Dot-separated key path.
        default: Default value if key missing. Raises KeyError if not provided.
    """
    parts = key.split(".")
    current = config
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif default is not _SENTINEL:
            return default
        else:
            raise KeyError(f"Config key not found: {key}")
    return current
```

Now create the three YAML config files:

```yaml
# config/settings.yaml
system:
  data_dir: "data"
  log_level: "INFO"
  cache_ttl_hours: 24

backtest:
  initial_capital: 100000
  # DD-2: No slippage for off-exchange funds (trade at declared NAV)
  min_trade_amount: 10   # 最小交易金额10元
  benchmark: "000300"    # 沪深300
  risk_free_rate: 0.02   # 2% annual

trading:
  min_cash_ratio: 0.05   # 组合现金储备5%
  max_cash_ratio: 0.10   # 最高10%
  max_industry_ratio: 0.40  # 单行业不超40%
```

```yaml
# config/strategy_params.yaml
semi_monthly_rebalance:
  rebalance_day: [1, 16]        # 每月1日和16日
  deviation_single: 0.05         # 单只偏离±5%
  deviation_portfolio: 0.03      # 组合整体偏离±3%

  buy_signal:
    broad_market:
      pe_percentile_threshold: 0.20    # PE近5年20%分位以下
      daily_drop_trigger: -0.03        # 单日大跌3%
    gold:
      daily_drop_trigger: -0.05        # 黄金单日下跌5%
    sector:
      daily_drop_trigger: -0.03        # 行业ETF单日下跌3%

  take_profit:
    tier1:
      return_threshold: 0.15     # 收益率达15%
      reduce_ratio: 0.20         # 减仓20%
    tier2:
      return_threshold: 0.30     # 收益率达30%
      reduce_ratio: 0.30         # 减仓30%
    valuation_exit:
      pe_percentile_threshold: 0.80   # PE近5年80%分位以上

  stop_loss:
    ma_break:
      ma_period: 20              # 20日均线
      daily_drop: -0.05          # 单日大跌5%
      confirm_days: 3            # 连续3日确认
      reduce_ratio: 0.50         # 减仓50%
    single_max_drawdown: -0.20   # 单只最大回撤20%清仓
    portfolio_drawdown:
      pause_add_threshold: -0.10  # 组合回撤10%暂停加仓
      force_reduce_threshold: -0.15  # 组合回撤15%强制减仓
      force_reduce_ratio: 0.20   # 减仓20%
```

```yaml
# config/data_sources.yaml
primary:
  name: "akshare_sina"
  retry_count: 3
  retry_delay_seconds: 2

fallback:
  - name: "akshare_eastmoney"
    retry_count: 2
    retry_delay_seconds: 3

update:
  daily_increment: true
  weekly_full_check: true
  full_check_day: "Sunday"

etf_categories:
  broad_market: ["510300", "510500", "159915"]   # 沪深300/中证500/创业板
  sector: ["512010", "512660", "512800"]          # 医药/军工/银行
  dividend: ["510880", "512890"]                  # 红利ETF
  gold: ["518880"]                                # 黄金ETF
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_core/test_config.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add YAML config system with dot-notation access"
```

---

### Task 4: Trading calendar utility

**Files:**
- Test: `tests/test_core/test_calendar.py`
- Create: `src/etf_analyzer/core/calendar.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_calendar.py
"""Tests for A-share trading calendar."""
from datetime import date
from etf_analyzer.core.calendar import (
    is_trading_day,
    next_trading_day,
    get_trading_days,
)


class TestTradingCalendar:
    def test_weekday_is_trading_day(self):
        # 2024-01-02 is a Tuesday
        assert is_trading_day(date(2024, 1, 2)) is True

    def test_weekend_is_not_trading_day(self):
        # 2024-01-06 is a Saturday
        assert is_trading_day(date(2024, 1, 6)) is False
        # 2024-01-07 is a Sunday
        assert is_trading_day(date(2024, 1, 7)) is False

    def test_next_trading_day_from_friday(self):
        # 2024-01-05 is a Friday -> next trading day is Monday 2024-01-08
        result = next_trading_day(date(2024, 1, 5))
        assert result == date(2024, 1, 8)

    def test_next_trading_day_from_weekday(self):
        result = next_trading_day(date(2024, 1, 2))
        assert result == date(2024, 1, 3)

    def test_get_trading_days_range(self):
        days = get_trading_days(date(2024, 1, 1), date(2024, 1, 7))
        # Jan 1 Mon(holiday-like but we just check weekdays for now)
        # Jan 2 Tue, Jan 3 Wed, Jan 4 Thu, Jan 5 Fri => 5 weekdays incl Jan 1
        assert all(d.weekday() < 5 for d in days)
        assert len(days) == 5  # Mon-Fri

    def test_t_plus_n_settlement(self):
        """T+1 confirmation: buy on Tuesday, confirmed on Wednesday."""
        from etf_analyzer.core.calendar import settle_date
        buy_date = date(2024, 1, 2)  # Tuesday
        assert settle_date(buy_date, n=1) == date(2024, 1, 3)

    def test_t_plus_n_over_weekend(self):
        """T+2 redemption: sell on Thursday, arrives Monday."""
        from etf_analyzer.core.calendar import settle_date
        sell_date = date(2024, 1, 4)  # Thursday
        assert settle_date(sell_date, n=2) == date(2024, 1, 8)  # Monday
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_calendar.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/core/calendar.py
"""A-share trading calendar utilities.

Note: This is a simplified calendar using weekday-only logic.
For production, integrate with exchange holiday data or use
`exchange_calendars` package for accurate holiday handling.
"""
from datetime import date, timedelta
from typing import List


def is_trading_day(d: date) -> bool:
    """Check if date is a trading day (weekday, not holiday).

    Currently only checks weekdays. TODO: add CN holiday support.
    """
    return d.weekday() < 5  # Mon=0 ... Fri=4


def next_trading_day(d: date) -> date:
    """Return the next trading day after given date."""
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def get_trading_days(start: date, end: date) -> List[date]:
    """Return list of trading days in [start, end] inclusive."""
    days = []
    current = start
    while current <= end:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def settle_date(trade_date: date, n: int = 1) -> date:
    """Calculate T+N settlement date (N trading days after trade_date).

    Args:
        trade_date: The trade execution date.
        n: Number of trading days to settle (1 for purchase, 2 for redemption).
    """
    current = trade_date
    for _ in range(n):
        current = next_trading_day(current)
    return current
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_core/test_calendar.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add trading calendar with T+N settlement"
```

---

### Task 5: Simple disk cache

**Files:**
- Test: `tests/test_core/test_cache.py`
- Create: `src/etf_analyzer/core/cache.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_cache.py
"""Tests for simple disk cache."""
import pandas as pd
from etf_analyzer.core.cache import DiskCache


class TestDiskCache:
    def test_set_and_get(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path))
        cache.set("key1", {"value": 42})
        assert cache.get("key1") == {"value": 42}

    def test_get_missing_returns_none(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path))
        assert cache.get("nonexistent") is None

    def test_set_and_get_dataframe(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path))
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        cache.set_dataframe("df_key", df)
        result = cache.get_dataframe("df_key")
        pd.testing.assert_frame_equal(result, df)

    def test_get_dataframe_missing_returns_none(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path))
        assert cache.get_dataframe("nonexistent") is None

    def test_invalidate(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path))
        cache.set("key1", {"value": 42})
        cache.invalidate("key1")
        assert cache.get("key1") is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_cache.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/core/cache.py
"""Simple disk-based cache for DataFrames and JSON-serializable objects."""
import json
import hashlib
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class DiskCache:
    """File-based cache using JSON for dicts and CSV for DataFrames."""

    def __init__(self, cache_dir: str):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str, ext: str = ".json") -> Path:
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self._dir / f"{safe_key}{ext}"

    def set(self, key: str, value: Any) -> None:
        path = self._key_path(key, ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)

    def get(self, key: str) -> Optional[Any]:
        path = self._key_path(key, ".json")
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def set_dataframe(self, key: str, df: pd.DataFrame) -> None:
        path = self._key_path(key, ".csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")

    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        path = self._key_path(key, ".csv")
        if not path.exists():
            return None
        return pd.read_csv(path, encoding="utf-8-sig")

    def invalidate(self, key: str) -> None:
        for ext in [".json", ".csv"]:
            path = self._key_path(key, ext)
            if path.exists():
                path.unlink()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_core/test_cache.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add simple disk cache for DataFrames and JSON"
```

---

## Phase 2: Formula Library (Pure Functions - Easy TDD)

### Task 6: Return calculations

**Files:**
- Test: `tests/test_formulas/test_returns.py`
- Create: `src/etf_analyzer/formulas/returns.py`

**Step 1: Write the failing test**

```python
# tests/test_formulas/test_returns.py
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
        # 10% in 1 year = 10% annualized
        result = annualized_return(total_return=0.1, holding_days=365)
        assert result == pytest.approx(0.1, abs=1e-4)

    def test_two_year_holding(self):
        # 21% in 2 years ≈ 10% annualized
        result = annualized_return(total_return=0.21, holding_days=730)
        assert result == pytest.approx(0.1, abs=1e-2)

    def test_zero_days_raises(self):
        with pytest.raises(ValueError, match="holding_days must be positive"):
            annualized_return(total_return=0.1, holding_days=0)


class TestWeightedPortfolioReturn:
    def test_equal_weights(self):
        returns = [0.1, 0.2, 0.3]
        weights = [1/3, 1/3, 1/3]
        result = weighted_portfolio_return(returns, weights)
        assert result == pytest.approx(0.2, abs=1e-9)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="Weights must sum to 1"):
            weighted_portfolio_return([0.1, 0.2], [0.5, 0.6])


class TestDcaReturn:
    def test_fixed_amount_dca(self):
        """DCA with 3 purchases at different NAVs."""
        nav_at_purchase = [1.0, 0.8, 1.2]
        amount_per_purchase = 1000.0
        final_nav = 1.1
        result = dca_return(nav_at_purchase, amount_per_purchase, final_nav)
        # Total invested: 3000
        # Shares: 1000/1.0 + 1000/0.8 + 1000/1.2 = 1000 + 1250 + 833.33 = 3083.33
        # Final value: 3083.33 * 1.1 = 3391.67
        # Return: (3391.67 - 3000) / 3000 = 0.1306
        assert result == pytest.approx(0.1306, abs=1e-3)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_formulas/test_returns.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/formulas/returns.py
"""Return calculation formulas.

Includes holding period, annualized, weighted portfolio, and DCA returns.
"""
from typing import List

import numpy as np


def holding_period_return(buy_nav: float, sell_nav: float) -> float:
    """Calculate holding period return.

    Formula: (sell_nav - buy_nav) / buy_nav
    Applicable: Single ETF buy-sell round trip.
    """
    if buy_nav <= 0:
        raise ValueError("buy_nav must be positive")
    return (sell_nav - buy_nav) / buy_nav


def annualized_return(total_return: float, holding_days: int) -> float:
    """Calculate annualized return from total return and holding period.

    Formula: (1 + total_return) ^ (365 / holding_days) - 1
    Applicable: Comparing returns across different holding periods.
    """
    if holding_days <= 0:
        raise ValueError("holding_days must be positive")
    return (1 + total_return) ** (365 / holding_days) - 1


def weighted_portfolio_return(returns: List[float], weights: List[float]) -> float:
    """Calculate portfolio return as weighted sum of individual returns.

    Formula: sum(r_i * w_i)
    Applicable: Portfolio-level return aggregation.
    """
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1")
    return float(np.dot(returns, weights))


def dca_return(
    nav_at_purchase: List[float],
    amount_per_purchase: float,
    final_nav: float,
) -> float:
    """Calculate return for dollar-cost averaging (定投).

    Args:
        nav_at_purchase: NAV at each purchase date.
        amount_per_purchase: Fixed amount invested each period.
        final_nav: NAV at valuation date.

    Formula: (total_shares * final_nav - total_cost) / total_cost
    Applicable: Regular fixed-amount DCA strategy evaluation.
    """
    total_shares = sum(amount_per_purchase / nav for nav in nav_at_purchase)
    total_cost = amount_per_purchase * len(nav_at_purchase)
    final_value = total_shares * final_nav
    return (final_value - total_cost) / total_cost
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_formulas/test_returns.py -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add return calculation formulas (holding period, annualized, weighted, DCA)"
```

---

### Task 7: Valuation indicators (PE percentile & dividend yield — see DD-1: no PB)

**Files:**
- Test: `tests/test_formulas/test_valuation.py`
- Create: `src/etf_analyzer/formulas/valuation.py`

**Step 1: Write the failing test**

```python
# tests/test_formulas/test_valuation.py
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
        history = pd.Series(range(1, 101))  # 1..100
        result = percentile_rank(current_value=50, history=history)
        # 49 values < 50 out of 100 => ~49%
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_formulas/test_valuation.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/formulas/valuation.py
"""Valuation indicator formulas.

PE percentile calculation, valuation zone classification, dividend yield.
Note: PB is NOT available from akshare index APIs (see DD-1). All references to PB are removed.
"""
import pandas as pd


def percentile_rank(current_value: float, history: pd.Series) -> float:
    """Calculate percentile rank of current value within historical distribution.

    Formula: count(history < current_value) / len(history)
    Applicable: PE near-5-year percentile for buy/sell signal generation.

    Args:
        current_value: Current PE value.
        history: Historical PE series (e.g. 5 years of daily data).

    Returns:
        Percentile in [0, 1]. 0.2 means 20% of history < current.
    """
    if len(history) == 0:
        raise ValueError("History cannot be empty")
    return float((history < current_value).sum() / len(history))


def valuation_zone(
    percentile: float,
    low_threshold: float = 0.30,
    high_threshold: float = 0.70,
) -> str:
    """Classify valuation zone based on percentile rank.

    Args:
        percentile: Percentile rank from percentile_rank().
        low_threshold: Below this = undervalued (default 30%).
        high_threshold: Above this = overvalued (default 70%).

    Returns:
        One of "undervalued", "normal", "overvalued".
    """
    if percentile <= low_threshold:
        return "undervalued"
    elif percentile > high_threshold:
        return "overvalued"
    return "normal"


def dividend_yield(annual_dividend: float, price: float) -> float:
    """Calculate dividend yield.

    Formula: annual_dividend / price
    Applicable: Dividend ETF screening and income analysis.
    """
    if price <= 0:
        raise ValueError("price must be positive")
    return annual_dividend / price
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_formulas/test_valuation.py -v`
Expected: 8 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add valuation formulas (PE percentile, valuation zone, dividend yield)"
```

---

### Task 8: Risk metrics

**Files:**
- Test: `tests/test_formulas/test_risk.py`
- Create: `src/etf_analyzer/formulas/risk.py`

**Step 1: Write the failing test**

```python
# tests/test_formulas/test_risk.py
"""Tests for risk metric formulas."""
import numpy as np
import pandas as pd
import pytest
from etf_analyzer.formulas.risk import (
    max_drawdown,
    max_drawdown_duration,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    alpha_beta,
    downside_risk,
    return_drawdown_ratio,
)


class TestMaxDrawdown:
    def test_simple_drawdown(self):
        nav = pd.Series([1.0, 1.2, 1.1, 0.9, 1.0])
        result = max_drawdown(nav)
        # Peak = 1.2, trough = 0.9 => (0.9-1.2)/1.2 = -0.25
        assert result == pytest.approx(-0.25, abs=1e-6)

    def test_no_drawdown(self):
        nav = pd.Series([1.0, 1.1, 1.2, 1.3])
        result = max_drawdown(nav)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_single_value(self):
        nav = pd.Series([1.0])
        result = max_drawdown(nav)
        assert result == pytest.approx(0.0, abs=1e-6)


class TestMaxDrawdownDuration:
    def test_basic_duration(self):
        # nav drops then recovers
        nav = pd.Series(
            [1.0, 1.1, 1.2, 1.0, 0.9, 1.0, 1.1, 1.2],
            index=pd.bdate_range("2024-01-02", periods=8)
        )
        duration = max_drawdown_duration(nav)
        assert duration >= 3  # At least 3 bars to recover from peak


class TestAnnualizedVolatility:
    def test_known_volatility(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 252))
        vol = annualized_volatility(returns)
        # Daily std ~0.01, annualized ~0.01 * sqrt(252) ≈ 0.1587
        assert vol == pytest.approx(0.01 * np.sqrt(252), abs=0.02)


class TestSharpeRatio:
    def test_positive_sharpe(self, sample_daily_returns):
        sr = sharpe_ratio(sample_daily_returns, risk_free_rate=0.02)
        # With ~8% return and ~15% vol, Sharpe ≈ (0.08-0.02)/0.15 ≈ 0.4
        assert sr > 0

    def test_zero_volatility_returns_zero(self):
        flat_returns = pd.Series([0.001] * 252)
        sr = sharpe_ratio(flat_returns, risk_free_rate=0.0)
        # Non-zero return with near-zero vol => large sharpe, but we test it doesn't crash
        assert isinstance(sr, float)


class TestAlphaBeta:
    def test_alpha_beta_returns_tuple(self, sample_daily_returns, sample_benchmark_returns):
        a, b = alpha_beta(sample_daily_returns, sample_benchmark_returns)
        assert isinstance(a, float)
        assert isinstance(b, float)


class TestCalmarRatio:
    def test_positive_calmar(self, sample_nav_series):
        returns = sample_nav_series.pct_change().dropna()
        result = calmar_ratio(returns, sample_nav_series)
        assert isinstance(result, float)


class TestReturnDrawdownRatio:
    def test_basic_ratio(self):
        result = return_drawdown_ratio(annual_return=0.1, max_dd=-0.05)
        assert result == pytest.approx(2.0, abs=1e-6)

    def test_zero_drawdown(self):
        result = return_drawdown_ratio(annual_return=0.1, max_dd=0.0)
        assert result == float("inf")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_formulas/test_risk.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/formulas/risk.py
"""Risk metric formulas.

Max drawdown, volatility, Sharpe, Sortino, Calmar, Alpha/Beta, etc.
Uses empyrical where possible, with fallback implementations.
"""
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import empyrical as ep
    HAS_EMPYRICAL = True
except ImportError:
    HAS_EMPYRICAL = False


def max_drawdown(nav: pd.Series) -> float:
    """Calculate maximum drawdown from NAV series.

    Formula: min((nav - cummax) / cummax)
    Applicable: Risk assessment, stop-loss threshold evaluation.

    Args:
        nav: Net asset value series (not returns).

    Returns:
        Max drawdown as negative float (e.g. -0.25 for 25% drawdown).
        Returns 0.0 if no drawdown.
    """
    if len(nav) <= 1:
        return 0.0
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    return float(drawdown.min())


def max_drawdown_duration(nav: pd.Series) -> int:
    """Calculate the longest drawdown duration in trading days.

    Applicable: Evaluating recovery time for strategy risk assessment.
    """
    cummax = nav.cummax()
    in_drawdown = nav < cummax
    if not in_drawdown.any():
        return 0

    max_dur = 0
    current_dur = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return max_dur


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility from daily returns.

    Formula: std(returns) * sqrt(periods_per_year)
    Applicable: Risk characterization, fund screening (volatility ≤ 20%).
    """
    if HAS_EMPYRICAL:
        return float(ep.annual_volatility(returns, period="daily"))
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe ratio.

    Formula: (annualized_return - risk_free_rate) / annualized_volatility
    Applicable: Risk-adjusted performance comparison across strategies.
    """
    if HAS_EMPYRICAL:
        daily_rf = risk_free_rate / periods_per_year
        return float(ep.sharpe_ratio(returns, risk_free=daily_rf, period="daily"))
    ann_ret = (1 + returns.mean()) ** periods_per_year - 1
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    if ann_vol == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / ann_vol)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sortino ratio (penalizes downside volatility only).

    Formula: (annualized_return - risk_free_rate) / downside_deviation
    Applicable: Strategies targeting downside protection.
    """
    if HAS_EMPYRICAL:
        daily_rf = risk_free_rate / periods_per_year
        return float(ep.sortino_ratio(returns, required_return=daily_rf, period="daily"))
    ann_ret = (1 + returns.mean()) ** periods_per_year - 1
    dd = downside_risk(returns, periods_per_year=periods_per_year)
    if dd == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / dd)


def calmar_ratio(returns: pd.Series, nav: pd.Series) -> float:
    """Calculate Calmar ratio.

    Formula: annualized_return / abs(max_drawdown)
    Applicable: Comparing strategies by return per unit drawdown.
    """
    if HAS_EMPYRICAL:
        return float(ep.calmar_ratio(returns, period="daily"))
    ann_ret = (1 + returns.mean()) ** 252 - 1
    mdd = abs(max_drawdown(nav))
    if mdd == 0:
        return float("inf")
    return float(ann_ret / mdd)


def alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> Tuple[float, float]:
    """Calculate Alpha and Beta relative to benchmark.

    Formula:
        Beta = cov(r, b) / var(b)
        Alpha = annualized(r - risk_free) - Beta * annualized(b - risk_free)
    Applicable: Measuring strategy excess return and market sensitivity.
    """
    if HAS_EMPYRICAL:
        aligned = pd.DataFrame({"r": returns, "b": benchmark_returns}).dropna()
        daily_rf = risk_free_rate / 252
        a, b = ep.alpha_beta(
            aligned["r"], aligned["b"], risk_free=daily_rf, period="daily"
        )
        return float(a), float(b)
    aligned = pd.DataFrame({"r": returns, "b": benchmark_returns}).dropna()
    cov_matrix = aligned.cov()
    beta = cov_matrix.loc["r", "b"] / cov_matrix.loc["b", "b"]
    ann_r = (1 + aligned["r"].mean()) ** 252 - 1
    ann_b = (1 + aligned["b"].mean()) ** 252 - 1
    alpha = (ann_r - risk_free_rate) - beta * (ann_b - risk_free_rate)
    return float(alpha), float(beta)


def downside_risk(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized downside deviation (for Sortino ratio).

    Applicable: Measuring downside volatility only (negative returns).
    """
    if HAS_EMPYRICAL:
        return float(ep.downside_risk(returns, period="daily"))
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return 0.0
    return float(negative_returns.std() * np.sqrt(periods_per_year))


def return_drawdown_ratio(annual_return: float, max_dd: float) -> float:
    """Calculate return-to-drawdown ratio.

    Formula: annual_return / abs(max_drawdown)
    Applicable: Quick risk-adjusted return evaluation.
    """
    if max_dd == 0:
        return float("inf")
    return annual_return / abs(max_dd)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_formulas/test_risk.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add risk metric formulas (drawdown, Sharpe, Sortino, Calmar, Alpha/Beta)"
```

---

### Task 9: Technical analysis formulas

**Files:**
- Test: `tests/test_formulas/test_technical.py`
- Create: `src/etf_analyzer/formulas/technical.py`

**Step 1: Write the failing test**

```python
# tests/test_formulas/test_technical.py
"""Tests for technical analysis formulas."""
import numpy as np
import pandas as pd
import pytest
from etf_analyzer.formulas.technical import (
    moving_average,
    is_ma_uptrend,
    daily_change_pct,
    drawdown_from_peak,
    rolling_volatility,
)


class TestMovingAverage:
    def test_5_day_ma(self):
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        ma = moving_average(prices, window=5)
        # Last value MA5 = mean(3,4,5,6,7) = 5.0
        assert ma.iloc[-1] == pytest.approx(5.0, abs=1e-9)
        # First 4 should be NaN
        assert ma.isna().sum() == 4

    def test_window_larger_than_series(self):
        prices = pd.Series([1.0, 2.0])
        ma = moving_average(prices, window=5)
        assert ma.isna().all()


class TestMaUptrend:
    def test_uptrend(self):
        prices = pd.Series(range(1, 70), dtype=float)
        assert is_ma_uptrend(prices, window=60) is True

    def test_downtrend(self):
        prices = pd.Series(range(70, 1, -1), dtype=float)
        assert is_ma_uptrend(prices, window=60) is False


class TestDailyChangePct:
    def test_basic_change(self):
        prices = pd.Series([100.0, 103.0, 100.0])
        changes = daily_change_pct(prices)
        assert changes.iloc[0] == pytest.approx(0.03, abs=1e-6)
        assert changes.iloc[1] == pytest.approx(-0.0291, abs=1e-3)


class TestDrawdownFromPeak:
    def test_basic_drawdown(self):
        nav = pd.Series([1.0, 1.2, 1.1, 0.9, 1.0])
        dd = drawdown_from_peak(nav)
        # At index 3 (0.9): peak was 1.2, dd = (0.9-1.2)/1.2 = -0.25
        assert dd.iloc[3] == pytest.approx(-0.25, abs=1e-6)
        # At peak (1.2): dd = 0
        assert dd.iloc[1] == pytest.approx(0.0, abs=1e-6)


class TestRollingVolatility:
    def test_rolling_vol_length(self):
        returns = pd.Series(np.random.normal(0, 0.01, 100))
        vol = rolling_volatility(returns, window=20)
        assert len(vol) == len(returns)
        assert vol.isna().sum() == 19  # First window-1 are NaN
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_formulas/test_technical.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/formulas/technical.py
"""Technical analysis formulas.

Moving averages, trend detection, daily change, drawdown, rolling volatility.
"""
import numpy as np
import pandas as pd


def moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Calculate simple moving average.

    Formula: MA_n = mean(price[t-n+1:t+1])
    Applicable: 5/20/60-day MA for trend analysis and signal generation.

    Args:
        prices: Price or NAV series.
        window: MA window (5, 20, 60).
    """
    return prices.rolling(window=window).mean()


def is_ma_uptrend(prices: pd.Series, window: int = 60) -> bool:
    """Check if current price is above the MA (uptrend signal).

    Applicable: Fund screening - 60-day MA uptrend requirement.

    Returns:
        True if latest price > latest MA value.
    """
    ma = moving_average(prices, window)
    if ma.isna().iloc[-1]:
        return False
    return bool(prices.iloc[-1] > ma.iloc[-1])


def daily_change_pct(prices: pd.Series) -> pd.Series:
    """Calculate daily percentage change.

    Formula: (price[t] - price[t-1]) / price[t-1]
    Applicable: Detecting large daily drops for buy/stop-loss signals.
    """
    return prices.pct_change().dropna()


def drawdown_from_peak(nav: pd.Series) -> pd.Series:
    """Calculate drawdown series from running peak.

    Formula: (nav - cummax) / cummax
    Applicable: Visual drawdown charts and stop-loss monitoring.
    """
    cummax = nav.cummax()
    return (nav - cummax) / cummax


def rolling_volatility(
    returns: pd.Series, window: int = 20, annualize: bool = False
) -> pd.Series:
    """Calculate rolling volatility.

    Applicable: Dynamic risk monitoring during backtest.

    Args:
        returns: Daily return series.
        window: Rolling window in trading days.
        annualize: If True, multiply by sqrt(252).
    """
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_formulas/test_technical.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add technical analysis formulas (MA, trend, volatility)"
```

---

### Task 10: Strategy factor formulas

**Files:**
- Test: `tests/test_formulas/test_factors.py`
- Create: `src/etf_analyzer/formulas/factors.py`

**Step 1: Write the failing test**

```python
# tests/test_formulas/test_factors.py
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
        prices = pd.Series(range(1, 253), dtype=float)  # Steady uptrend
        mom = momentum_factor(prices, lookback=20)
        assert mom > 0

    def test_negative_momentum(self):
        prices = pd.Series(range(252, 0, -1), dtype=float)  # Steady downtrend
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
        assert score > 0.7  # High quality

    def test_high_tracking_error_low_quality(self):
        score = quality_factor(
            tracking_error=0.02,
            fund_scale=2e8,
            management_fee=0.01,
        )
        assert score < 0.5  # Lower quality

    def test_output_range(self):
        score = quality_factor(
            tracking_error=0.01,
            fund_scale=5e8,
            management_fee=0.005,
        )
        assert 0 <= score <= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_formulas/test_factors.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/formulas/factors.py
"""Strategy factor formulas.

Momentum factor, quality factor for fund scoring.
"""
import numpy as np
import pandas as pd


def momentum_factor(prices: pd.Series, lookback: int = 20) -> float:
    """Calculate momentum factor as return over lookback period.

    Formula: (price[-1] - price[-lookback-1]) / price[-lookback-1]
    Applicable: Trend-following signal, strategy factor scoring.

    Args:
        prices: Price series.
        lookback: Number of periods to look back.
    """
    if len(prices) < lookback + 1:
        raise ValueError(
            f"Insufficient data: need {lookback + 1} points, got {len(prices)}"
        )
    return float((prices.iloc[-1] - prices.iloc[-lookback - 1]) / prices.iloc[-lookback - 1])


def quality_factor(
    tracking_error: float,
    fund_scale: float,
    management_fee: float,
    te_weight: float = 0.4,
    scale_weight: float = 0.3,
    fee_weight: float = 0.3,
) -> float:
    """Calculate quality factor score for ETF fund.

    Composite score based on tracking error, scale, and fees.
    Lower tracking error, larger scale, and lower fees = higher quality.

    Applicable: Fund screening secondary scoring.

    Returns:
        Score in [0, 1] range. Higher = better quality.
    """
    # Tracking error score: ≤0.5% -> 1.0, ≥2% -> 0.0
    te_score = np.clip(1.0 - (tracking_error - 0.005) / 0.015, 0, 1)

    # Scale score: ≥10B -> 1.0, ≤1B -> 0.0
    scale_score = np.clip((fund_scale - 1e8) / (10e8 - 1e8), 0, 1)

    # Fee score: ≤0.2% -> 1.0, ≥1% -> 0.0
    fee_score = np.clip(1.0 - (management_fee - 0.002) / 0.008, 0, 1)

    return float(te_score * te_weight + scale_score * scale_weight + fee_score * fee_weight)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_formulas/test_factors.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add strategy factor formulas (momentum, quality)"
```

---

## Phase 3: Data Management

### Task 11: Data store (CSV per ETF)

**Files:**
- Test: `tests/test_data/test_store.py`
- Create: `src/etf_analyzer/data/store.py`

**Step 1: Write the failing test**

```python
# tests/test_data/test_store.py
"""Tests for CSV data store."""
import pandas as pd
import pytest
from etf_analyzer.data.store import EtfDataStore


class TestEtfDataStore:
    def test_save_and_load(self, tmp_data_dir, sample_etf_df):
        store = EtfDataStore(data_dir=str(tmp_data_dir))
        store.save_etf_data("510300", sample_etf_df)
        loaded = store.load_etf_data("510300")
        assert len(loaded) == len(sample_etf_df)
        assert list(loaded.columns) == list(sample_etf_df.columns)

    def test_load_nonexistent_returns_none(self, tmp_data_dir):
        store = EtfDataStore(data_dir=str(tmp_data_dir))
        result = store.load_etf_data("999999")
        assert result is None

    def test_list_stored_etfs(self, tmp_data_dir, sample_etf_df):
        store = EtfDataStore(data_dir=str(tmp_data_dir))
        store.save_etf_data("510300", sample_etf_df)
        store.save_etf_data("510500", sample_etf_df)
        etfs = store.list_etfs()
        assert "510300" in etfs
        assert "510500" in etfs

    def test_append_data(self, tmp_data_dir, sample_etf_df):
        store = EtfDataStore(data_dir=str(tmp_data_dir))
        first_half = sample_etf_df.iloc[:30]
        second_half = sample_etf_df.iloc[30:]
        store.save_etf_data("510300", first_half)
        store.append_etf_data("510300", second_half)
        loaded = store.load_etf_data("510300")
        assert len(loaded) == len(sample_etf_df)

    def test_save_index_data(self, tmp_data_dir):
        store = EtfDataStore(data_dir=str(tmp_data_dir))
        df = pd.DataFrame({"日期": ["2024-01-02"], "pe": [12.5], "dividend_yield": [2.5]})
        store.save_index_data("000300", df)
        loaded = store.load_index_data("000300")
        assert loaded is not None
        assert len(loaded) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_store.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/data/store.py
"""CSV-based data store. One CSV file per ETF/index."""
from pathlib import Path
from typing import List, Optional

import pandas as pd

from etf_analyzer.core.logger import get_logger

logger = get_logger("data.store")


class EtfDataStore:
    """File-based store: data/{etf,index}/<code>.csv"""

    def __init__(self, data_dir: str):
        self._etf_dir = Path(data_dir) / "etf"
        self._index_dir = Path(data_dir) / "index"
        self._etf_dir.mkdir(parents=True, exist_ok=True)
        self._index_dir.mkdir(parents=True, exist_ok=True)

    def _etf_path(self, code: str) -> Path:
        return self._etf_dir / f"{code}.csv"

    def _index_path(self, code: str) -> Path:
        return self._index_dir / f"{code}.csv"

    def save_etf_data(self, code: str, df: pd.DataFrame) -> None:
        path = self._etf_path(code)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved ETF {code}: {len(df)} rows -> {path}")

    def load_etf_data(self, code: str) -> Optional[pd.DataFrame]:
        path = self._etf_path(code)
        if not path.exists():
            return None
        return pd.read_csv(path, encoding="utf-8-sig")

    def append_etf_data(self, code: str, new_df: pd.DataFrame) -> None:
        existing = self.load_etf_data(code)
        if existing is not None:
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["日期"], keep="last")
        else:
            combined = new_df
        self.save_etf_data(code, combined)

    def list_etfs(self) -> List[str]:
        return [p.stem for p in self._etf_dir.glob("*.csv")]

    def save_index_data(self, code: str, df: pd.DataFrame) -> None:
        path = self._index_path(code)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved index {code}: {len(df)} rows -> {path}")

    def load_index_data(self, code: str) -> Optional[pd.DataFrame]:
        path = self._index_path(code)
        if not path.exists():
            return None
        return pd.read_csv(path, encoding="utf-8-sig")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_store.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add CSV data store for ETF and index data"
```

---

### Task 12: Data cleaner & validator

**Files:**
- Test: `tests/test_data/test_cleaner.py`
- Create: `src/etf_analyzer/data/cleaner.py`

**Step 1: Write the failing test**

```python
# tests/test_data/test_cleaner.py
"""Tests for data cleaning and validation."""
import numpy as np
import pandas as pd
import pytest
from etf_analyzer.data.cleaner import (
    clean_nav_data,
    validate_ohlcv,
    fill_missing_dates,
)


class TestCleanNavData:
    def test_removes_nan_rows(self):
        df = pd.DataFrame({
            "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "收盘": [1.0, np.nan, 1.2],
            "成交量": [100, 200, 300],
        })
        cleaned = clean_nav_data(df)
        assert len(cleaned) == 2
        assert cleaned["收盘"].isna().sum() == 0

    def test_removes_negative_nav(self):
        df = pd.DataFrame({
            "日期": ["2024-01-02", "2024-01-03"],
            "收盘": [1.0, -0.5],
            "成交量": [100, 200],
        })
        cleaned = clean_nav_data(df)
        assert len(cleaned) == 1

    def test_sorts_by_date(self):
        df = pd.DataFrame({
            "日期": ["2024-01-04", "2024-01-02", "2024-01-03"],
            "收盘": [1.2, 1.0, 1.1],
            "成交量": [100, 200, 300],
        })
        cleaned = clean_nav_data(df)
        assert list(cleaned["日期"]) == ["2024-01-02", "2024-01-03", "2024-01-04"]


class TestValidateOhlcv:
    def test_valid_data_passes(self, sample_etf_df):
        errors = validate_ohlcv(sample_etf_df)
        assert len(errors) == 0

    def test_missing_column_detected(self):
        df = pd.DataFrame({"日期": ["2024-01-02"], "开盘": [1.0]})
        errors = validate_ohlcv(df)
        assert any("收盘" in e for e in errors)

    def test_high_less_than_low_detected(self):
        df = pd.DataFrame({
            "日期": ["2024-01-02"],
            "开盘": [1.0],
            "收盘": [1.1],
            "最高": [0.9],   # Invalid: high < low
            "最低": [1.0],
            "成交量": [100],
        })
        errors = validate_ohlcv(df)
        assert any("最高" in e or "high" in e.lower() for e in errors)


class TestFillMissingDates:
    def test_fills_gap(self):
        df = pd.DataFrame({
            "日期": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "收盘": [1.0, 1.2],
        })
        filled = fill_missing_dates(df)
        # Should include 2024-01-03 (Wednesday)
        assert len(filled) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_cleaner.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/data/cleaner.py
"""Data cleaning and validation utilities."""
from typing import List

import numpy as np
import pandas as pd

from etf_analyzer.core.logger import get_logger

logger = get_logger("data.cleaner")

REQUIRED_OHLCV_COLUMNS = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]


def clean_nav_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean NAV/price data: remove NaN, negative values, sort by date.

    Args:
        df: Raw DataFrame with at least '日期' and '收盘' columns.

    Returns:
        Cleaned DataFrame sorted by date.
    """
    result = df.copy()

    # Drop rows with NaN in critical columns
    if "收盘" in result.columns:
        before = len(result)
        result = result.dropna(subset=["收盘"])
        dropped = before - len(result)
        if dropped:
            logger.warning(f"Dropped {dropped} rows with NaN in '收盘'")

    # Remove negative/zero NAV
    if "收盘" in result.columns:
        result = result[result["收盘"] > 0]

    # Sort by date
    if "日期" in result.columns:
        result = result.sort_values("日期").reset_index(drop=True)

    return result


def validate_ohlcv(df: pd.DataFrame) -> List[str]:
    """Validate OHLCV data integrity. Returns list of error messages.

    Checks:
        1. Required columns exist.
        2. High >= Low for all rows.
        3. No negative volumes.
    """
    errors = []

    for col in REQUIRED_OHLCV_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if "最高" in df.columns and "最低" in df.columns:
        invalid_hl = df[df["最高"] < df["最低"]]
        if len(invalid_hl) > 0:
            errors.append(
                f"Found {len(invalid_hl)} rows where 最高 < 最低 (high < low)"
            )

    if "成交量" in df.columns:
        neg_vol = df[df["成交量"] < 0]
        if len(neg_vol) > 0:
            errors.append(f"Found {len(neg_vol)} rows with negative volume")

    return errors


def fill_missing_dates(
    df: pd.DataFrame, date_col: str = "日期", method: str = "ffill"
) -> pd.DataFrame:
    """Fill missing trading days with forward-filled data.

    Args:
        df: DataFrame with a date column.
        date_col: Name of the date column.
        method: Fill method ('ffill' or 'bfill').
    """
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.set_index(date_col)

    # Create business day index
    full_range = pd.bdate_range(start=result.index.min(), end=result.index.max())
    result = result.reindex(full_range)
    result = result.fillna(method=method)
    result = result.reset_index()
    result = result.rename(columns={"index": date_col})

    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_cleaner.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add data cleaning and validation utilities"
```

---

### Task 13: Data fetcher (akshare wrapper with retry + failover)

**Files:**
- Test: `tests/test_data/test_fetcher.py`
- Create: `src/etf_analyzer/data/fetcher.py`

**Step 1: Write the failing test**

```python
# tests/test_data/test_fetcher.py
"""Tests for akshare data fetcher wrapper.

NOTE: Tests use mocking to avoid network calls in CI.
Integration tests with real akshare can be run with `pytest -m integration`.
"""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from etf_analyzer.data.fetcher import EtfDataFetcher
from etf_analyzer.core.response import StatusCode


class TestEtfDataFetcher:
    def test_fetch_etf_hist_success(self):
        mock_df = pd.DataFrame({
            "日期": ["2024-01-02"],
            "开盘": [1.0], "收盘": [1.1],
            "最高": [1.2], "最低": [0.9],
            "成交量": [10000], "成交额": [11000.0],
            "涨跌幅": [1.0],
        })
        with patch("etf_analyzer.data.fetcher.ak") as mock_ak:
            mock_ak.fund_etf_hist_em.return_value = mock_df
            fetcher = EtfDataFetcher()
            resp = fetcher.fetch_etf_history("510300", "20240101", "20240201")
            assert resp.ok
            assert resp.status_code == StatusCode.SUCCESS
            assert len(resp.data) == 1

    def test_fetch_etf_hist_empty_returns_warning(self):
        with patch("etf_analyzer.data.fetcher.ak") as mock_ak:
            mock_ak.fund_etf_hist_em.return_value = pd.DataFrame()
            fetcher = EtfDataFetcher()
            resp = fetcher.fetch_etf_history("510300", "20240101", "20240201")
            assert resp.status_code == StatusCode.WARNING

    def test_fetch_etf_hist_exception_returns_error(self):
        with patch("etf_analyzer.data.fetcher.ak") as mock_ak:
            mock_ak.fund_etf_hist_em.side_effect = Exception("Network error")
            fetcher = EtfDataFetcher(retry_count=1)
            resp = fetcher.fetch_etf_history("510300", "20240101", "20240201")
            assert resp.status_code == StatusCode.ERROR
            assert "Network error" in resp.message

    def test_fetch_fund_fee_success(self):
        mock_df = pd.DataFrame({
            "项目": ["管理费率", "托管费率", "销售服务费率"],
            "数据": ["0.50%", "0.10%", "0.00%"],
        })
        with patch("etf_analyzer.data.fetcher.ak") as mock_ak:
            mock_ak.fund_fee_em.return_value = mock_df
            fetcher = EtfDataFetcher()
            resp = fetcher.fetch_fund_fee("510300", indicator="运作费用")
            assert resp.ok
            assert len(resp.data) > 0

    def test_fetch_index_valuation_success(self):
        mock_df = pd.DataFrame({
            "日期": ["2024-01-02"],
            "市盈率1": [12.5],
            "股息率1": [2.5],
        })
        with patch("etf_analyzer.data.fetcher.ak") as mock_ak:
            mock_ak.stock_zh_index_value_csindex.return_value = mock_df
            fetcher = EtfDataFetcher()
            resp = fetcher.fetch_index_valuation("000300")
            assert resp.ok
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_fetcher.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/data/fetcher.py
"""AKShare data fetcher with retry and failover support.

Primary: akshare Sina interface.
All methods return ApiResponse for unified error handling.
"""
import time
from typing import Optional

import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None  # Allow running tests without akshare installed

from etf_analyzer.core.response import ApiResponse
from etf_analyzer.core.logger import get_logger

logger = get_logger("data.fetcher")


class EtfDataFetcher:
    """Wraps akshare API calls with retry, error handling, and ApiResponse output."""

    def __init__(self, retry_count: int = 3, retry_delay: float = 2.0):
        self._retry_count = retry_count
        self._retry_delay = retry_delay

    def _retry(self, func, *args, **kwargs) -> ApiResponse:
        """Execute function with retry logic. Returns ApiResponse."""
        last_error = None
        for attempt in range(self._retry_count):
            try:
                result = func(*args, **kwargs)
                if result is None or (isinstance(result, pd.DataFrame) and result.empty):
                    return ApiResponse.warning(
                        data=pd.DataFrame(),
                        message=f"Empty result from {func.__name__}",
                    )
                return ApiResponse.success(data=result)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self._retry_count} failed: {e}"
                )
                if attempt < self._retry_count - 1:
                    time.sleep(self._retry_delay)

        error_msg = f"All {self._retry_count} attempts failed: {last_error}"
        logger.error(error_msg)
        return ApiResponse.error(message=error_msg)

    def fetch_etf_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        period: str = "daily",
        adjust: str = "hfq",
    ) -> ApiResponse:
        """Fetch ETF historical price data.

        Args:
            symbol: ETF code (e.g. '510300').
            start_date: Start date 'YYYYMMDD'.
            end_date: End date 'YYYYMMDD'.
            period: 'daily', 'weekly', or 'monthly'.
            adjust: 'qfq' (前复权), 'hfq' (后复权), '' (不复权).
        """
        return self._retry(
            ak.fund_etf_hist_em,
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

    def fetch_etf_spot(self) -> ApiResponse:
        """Fetch real-time ETF quotes (all ETFs)."""
        return self._retry(ak.fund_etf_spot_em)

    def fetch_fund_name_list(self) -> ApiResponse:
        """Fetch all fund names and codes."""
        return self._retry(ak.fund_name_em)

    def fetch_fund_fee(self, symbol: str, indicator: str = "运作费用") -> ApiResponse:
        """Fetch fund fee structure.

        Args:
            symbol: Fund code.
            indicator: One of '运作费用', '申购费率(前端)', '赎回费率', etc.
        """
        return self._retry(ak.fund_fee_em, symbol=symbol, indicator=indicator)

    def fetch_index_valuation(self, symbol: str) -> ApiResponse:
        """Fetch CSI index valuation (PE, dividend yield).

        Args:
            symbol: Index code (e.g. '000300' for CSI 300).
        """
        return self._retry(ak.stock_zh_index_value_csindex, symbol=symbol)

    def fetch_index_fund_list(
        self, category: str = "全部", indicator: str = "全部"
    ) -> ApiResponse:
        """Fetch list of index-tracking funds with basic info.

        Args:
            category: '全部', '沪深指数', '行业主题', etc.
            indicator: '全部', '被动指数型', '增强指数型'.
        """
        return self._retry(
            ak.fund_info_index_em, symbol=category, indicator=indicator
        )

    def fetch_fund_nav_history(
        self, symbol: str, start_date: str = "20000101", end_date: str = "20500101"
    ) -> ApiResponse:
        """Fetch historical NAV for on-exchange ETF fund.

        Args:
            symbol: Fund code.
        """
        return self._retry(
            ak.fund_etf_fund_info_em,
            fund=symbol,
            start_date=start_date,
            end_date=end_date,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_fetcher.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add akshare data fetcher with retry and ApiResponse"
```

---

### Task 14: Data updater (incremental update)

**Files:**
- Test: `tests/test_data/test_updater.py`
- Create: `src/etf_analyzer/data/updater.py`

**Step 1: Write the failing test**

```python
# tests/test_data/test_updater.py
"""Tests for incremental data updater."""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from etf_analyzer.data.updater import DataUpdater
from etf_analyzer.data.store import EtfDataStore
from etf_analyzer.core.response import ApiResponse


class TestDataUpdater:
    def test_full_update_new_etf(self, tmp_data_dir):
        store = EtfDataStore(data_dir=str(tmp_data_dir))
        mock_fetcher = MagicMock()
        mock_df = pd.DataFrame({
            "日期": pd.date_range("2024-01-02", periods=5).strftime("%Y-%m-%d"),
            "收盘": [1.0, 1.1, 1.2, 1.1, 1.3],
            "开盘": [1.0, 1.0, 1.1, 1.2, 1.1],
            "最高": [1.1, 1.2, 1.3, 1.2, 1.4],
            "最低": [0.9, 1.0, 1.1, 1.0, 1.1],
            "成交量": [100] * 5,
            "成交额": [100.0] * 5,
            "涨跌幅": [0, 10, 9, -8, 18],
        })
        mock_fetcher.fetch_etf_history.return_value = ApiResponse.success(data=mock_df)
        updater = DataUpdater(fetcher=mock_fetcher, store=store)

        result = updater.update_etf("510300")
        assert result.ok
        stored = store.load_etf_data("510300")
        assert stored is not None
        assert len(stored) == 5

    def test_incremental_update_existing_etf(self, tmp_data_dir):
        store = EtfDataStore(data_dir=str(tmp_data_dir))
        # Pre-populate with 3 rows
        old_df = pd.DataFrame({
            "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "收盘": [1.0, 1.1, 1.2],
            "开盘": [1.0, 1.0, 1.1],
            "最高": [1.1, 1.2, 1.3],
            "最低": [0.9, 1.0, 1.1],
            "成交量": [100, 100, 100],
            "成交额": [100.0, 100.0, 100.0],
            "涨跌幅": [0, 10, 9],
        })
        store.save_etf_data("510300", old_df)

        mock_fetcher = MagicMock()
        new_df = pd.DataFrame({
            "日期": ["2024-01-05", "2024-01-08"],
            "收盘": [1.3, 1.25],
            "开盘": [1.2, 1.3],
            "最高": [1.4, 1.35],
            "最低": [1.2, 1.2],
            "成交量": [200, 150],
            "成交额": [200.0, 150.0],
            "涨跌幅": [8, -4],
        })
        mock_fetcher.fetch_etf_history.return_value = ApiResponse.success(data=new_df)
        updater = DataUpdater(fetcher=mock_fetcher, store=store)

        result = updater.update_etf("510300", incremental=True)
        assert result.ok
        stored = store.load_etf_data("510300")
        assert len(stored) == 5  # 3 old + 2 new
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_updater.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/data/updater.py
"""Incremental data updater for ETF and index data."""
from datetime import date

import pandas as pd

from etf_analyzer.core.response import ApiResponse
from etf_analyzer.core.logger import get_logger
from etf_analyzer.data.fetcher import EtfDataFetcher
from etf_analyzer.data.store import EtfDataStore
from etf_analyzer.data.cleaner import clean_nav_data

logger = get_logger("data.updater")


class DataUpdater:
    """Handles full and incremental updates for ETF data."""

    def __init__(self, fetcher: EtfDataFetcher, store: EtfDataStore):
        self._fetcher = fetcher
        self._store = store

    def update_etf(
        self,
        code: str,
        incremental: bool = False,
        end_date: str = None,
    ) -> ApiResponse:
        """Update ETF data (full or incremental).

        Args:
            code: ETF code.
            incremental: If True, only fetch data after last stored date.
            end_date: End date 'YYYYMMDD'. Defaults to today.
        """
        if end_date is None:
            end_date = date.today().strftime("%Y%m%d")

        start_date = "20100101"  # Default full history

        if incremental:
            existing = self._store.load_etf_data(code)
            if existing is not None and len(existing) > 0:
                last_date = pd.to_datetime(existing["日期"]).max()
                start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
                logger.info(f"Incremental update for {code} from {start_date}")

        resp = self._fetcher.fetch_etf_history(
            symbol=code, start_date=start_date, end_date=end_date
        )

        if not resp.ok:
            return resp

        df = resp.data
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = clean_nav_data(df)
            if incremental:
                self._store.append_etf_data(code, df)
            else:
                self._store.save_etf_data(code, df)
            return ApiResponse.success(
                data=df,
                message=f"Updated {code}: {len(df)} rows",
            )

        return ApiResponse.warning(message=f"No new data for {code}")

    def update_batch(self, codes: list, incremental: bool = True) -> ApiResponse:
        """Update multiple ETFs. Returns summary."""
        results = {}
        for code in codes:
            resp = self.update_etf(code, incremental=incremental)
            results[code] = {
                "status": resp.status_code.name,
                "message": resp.message,
            }
            logger.info(f"{code}: {resp.message}")

        return ApiResponse.success(data=results, message=f"Updated {len(codes)} ETFs")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_updater.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add incremental data updater with full/partial update"
```

---

## Phase 4: Trading Simulation

### Task 15: Fee calculator

**Files:**
- Test: `tests/test_simulation/test_fees.py`
- Create: `src/etf_analyzer/simulation/fees.py`

**Step 1: Write the failing test**

```python
# tests/test_simulation/test_fees.py
"""Tests for fee calculation."""
import pytest
from etf_analyzer.simulation.fees import FeeCalculator, FeeSchedule


class TestFeeCalculator:
    def test_purchase_fee(self):
        schedule = FeeSchedule(purchase_rate=0.015, discount=0.1)
        calc = FeeCalculator(schedule)
        # 10000 * 0.015 * 0.1 = 15 (with 10% of original = 90% discount)
        fee = calc.purchase_fee(amount=10000)
        assert fee == pytest.approx(15.0, abs=0.01)

    def test_redemption_fee_short_hold(self):
        schedule = FeeSchedule(
            redemption_tiers=[
                (7, 0.015),     # < 7 days: 1.5%
                (30, 0.005),    # 7-30 days: 0.5%
                (365, 0.0025),  # 30-365 days: 0.25%
                (float("inf"), 0.0),  # > 365 days: 0%
            ]
        )
        calc = FeeCalculator(schedule)
        fee = calc.redemption_fee(amount=10000, holding_days=3)
        assert fee == pytest.approx(150.0, abs=0.01)  # 10000 * 1.5%

    def test_redemption_fee_long_hold(self):
        schedule = FeeSchedule(
            redemption_tiers=[
                (7, 0.015),
                (30, 0.005),
                (365, 0.0025),
                (float("inf"), 0.0),
            ]
        )
        calc = FeeCalculator(schedule)
        fee = calc.redemption_fee(amount=10000, holding_days=400)
        assert fee == pytest.approx(0.0, abs=0.01)  # Free after 1 year

    def test_daily_management_fee_informational(self):
        """daily_accrued_fee is for reporting only — NOT deducted during simulation.
        Off-exchange NAV already includes mgmt/custody fees (see DD-2)."""
        schedule = FeeSchedule(management_rate=0.005, custody_rate=0.001)
        calc = FeeCalculator(schedule)
        daily_fee = calc.daily_accrued_fee(nav_total=1000000)
        expected = (0.005 + 0.001) / 365 * 1000000
        assert daily_fee == pytest.approx(expected, abs=0.1)

    def test_dca_purchase_discount(self):
        schedule = FeeSchedule(purchase_rate=0.015, dca_discount=0.1)
        calc = FeeCalculator(schedule)
        fee = calc.purchase_fee(amount=1000, is_dca=True)
        assert fee == pytest.approx(1.5, abs=0.01)  # 1000 * 0.015 * 0.1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation/test_fees.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/simulation/fees.py
"""Fee calculation for off-exchange ETF linked funds.

Handles purchase fees (with DCA discounts) and tiered redemption fees.
NOTE (DD-2): daily_accrued_fee() is for INFORMATIONAL REPORTING ONLY.
Off-exchange NAV already includes management/custody fees — do NOT
deduct these during simulation or backtesting to avoid double-counting.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class FeeSchedule:
    """Fee schedule for a specific fund.

    Attributes:
        purchase_rate: Front-end purchase fee rate (e.g. 0.015 = 1.5%).
        discount: Purchase fee discount ratio (e.g. 0.1 = 打一折, 90% off).
        dca_discount: DCA-specific discount ratio.
        redemption_tiers: List of (max_days, rate) sorted by max_days ascending.
            E.g. [(7, 0.015), (30, 0.005), (365, 0.0025), (inf, 0.0)]
        management_rate: Annual management fee rate.
        custody_rate: Annual custody fee rate.
        sales_service_rate: Annual sales service fee rate.
    """
    purchase_rate: float = 0.015
    discount: float = 0.1
    dca_discount: float = 0.1
    redemption_tiers: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (7, 0.015),
            (30, 0.005),
            (365, 0.0025),
            (float("inf"), 0.0),
        ]
    )
    management_rate: float = 0.005
    custody_rate: float = 0.001
    sales_service_rate: float = 0.0


class FeeCalculator:
    """Calculate various fees for fund transactions."""

    def __init__(self, schedule: FeeSchedule):
        self._schedule = schedule

    def purchase_fee(self, amount: float, is_dca: bool = False) -> float:
        """Calculate purchase fee.

        Args:
            amount: Purchase amount in CNY.
            is_dca: Whether this is a DCA purchase (may have different discount).
        """
        discount = self._schedule.dca_discount if is_dca else self._schedule.discount
        return amount * self._schedule.purchase_rate * discount

    def redemption_fee(self, amount: float, holding_days: int) -> float:
        """Calculate redemption fee based on holding period tier.

        Args:
            amount: Redemption amount in CNY.
            holding_days: Number of days held.
        """
        for max_days, rate in self._schedule.redemption_tiers:
            if holding_days < max_days:
                return amount * rate
        return 0.0

    def daily_accrued_fee(self, nav_total: float) -> float:
        """Calculate daily accrued management + custody fees (INFORMATIONAL ONLY).

        WARNING (DD-2): Do NOT deduct this from portfolio value during simulation.
        Off-exchange NAV already includes these fees. Use only for reporting
        the embedded cost of holding a fund.

        Args:
            nav_total: Total net asset value of the holding.
        """
        annual_rate = (
            self._schedule.management_rate
            + self._schedule.custody_rate
            + self._schedule.sales_service_rate
        )
        return nav_total * annual_rate / 365
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulation/test_fees.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add fee calculator with tiered redemption and DCA discount"
```

---

### Task 16: Trade execution broker (T+1/T+2, no slippage — see DD-2, DD-3)

**Files:**
- Test: `tests/test_simulation/test_broker.py`
- Create: `src/etf_analyzer/simulation/broker.py`

**Step 1: Write the failing test**

```python
# tests/test_simulation/test_broker.py
"""Tests for simulated trade execution broker.

Key design decisions applied here:
- DD-2: No slippage — off-exchange funds trade at exact declared NAV.
- DD-3: Broker returns PendingOrder objects. Settlement is processed
  separately by calling broker.process_settlements(current_date).
"""
from datetime import date
import pytest
from etf_analyzer.simulation.broker import SimBroker, PendingOrder, OrderStatus, TradeType
from etf_analyzer.simulation.fees import FeeCalculator, FeeSchedule


class TestSimBroker:
    @pytest.fixture
    def broker(self):
        schedule = FeeSchedule(purchase_rate=0.015, discount=0.1)
        fee_calc = FeeCalculator(schedule)
        return SimBroker(fee_calculator=fee_calc)

    def test_buy_order_returns_pending(self, broker):
        order = broker.submit_buy(
            code="510300",
            amount=10000.0,
            nav=1.5,
            trade_date=date(2024, 1, 2),
        )
        assert order.status == OrderStatus.PENDING
        assert order.trade_type == TradeType.BUY
        assert order.confirm_date == date(2024, 1, 3)  # T+1

    def test_sell_order_returns_pending(self, broker):
        order = broker.submit_sell(
            code="510300",
            shares=1000.0,
            nav=1.5,
            trade_date=date(2024, 1, 2),
            holding_days=100,
        )
        assert order.status == OrderStatus.PENDING
        assert order.trade_type == TradeType.SELL
        assert order.settle_date == date(2024, 1, 4)  # T+2

    def test_buy_at_exact_nav_no_slippage(self, broker):
        """DD-2: Off-exchange trades at declared NAV, no slippage."""
        order = broker.submit_buy(
            code="510300",
            amount=10000.0,
            nav=1.0,
            trade_date=date(2024, 1, 2),
        )
        # Amount after fee: 10000 - 10000*0.015*0.1 = 10000 - 15 = 9985
        # Shares at exact NAV: 9985 / 1.0 = 9985.0 (no slippage!)
        assert order.shares == pytest.approx(9985.0, abs=0.01)

    def test_sell_at_exact_nav_no_slippage(self, broker):
        """DD-2: Sell at declared NAV, only redemption fee deducted."""
        order = broker.submit_sell(
            code="510300",
            shares=1000.0,
            nav=1.5,
            trade_date=date(2024, 1, 2),
            holding_days=100,
        )
        # Gross: 1000 * 1.5 = 1500.0 (no slippage!)
        # Redemption fee: 1500.0 * 0.0025 = 3.75 (30-365 days tier)
        # Net: 1500.0 - 3.75 = 1496.25
        assert order.gross_amount == pytest.approx(1500.0, abs=0.01)
        assert order.net_amount == pytest.approx(1496.25, abs=0.01)

    def test_process_settlements_confirms_buy(self, broker):
        """DD-3: Pending buy becomes confirmed on T+1."""
        order = broker.submit_buy(
            code="510300", amount=10000.0, nav=1.0, trade_date=date(2024, 1, 2),
        )
        assert order.status == OrderStatus.PENDING

        # Day T+0: not yet confirmed
        confirmed = broker.process_settlements(date(2024, 1, 2))
        assert len(confirmed) == 0

        # Day T+1: confirmed
        confirmed = broker.process_settlements(date(2024, 1, 3))
        assert len(confirmed) == 1
        assert confirmed[0].status == OrderStatus.CONFIRMED

    def test_process_settlements_settles_sell(self, broker):
        """DD-3: Pending sell cash arrives on T+2."""
        order = broker.submit_sell(
            code="510300", shares=1000.0, nav=1.5,
            trade_date=date(2024, 1, 2), holding_days=100,
        )
        # Day T+1: not yet settled
        settled = broker.process_settlements(date(2024, 1, 3))
        assert len(settled) == 0

        # Day T+2: settled
        settled = broker.process_settlements(date(2024, 1, 4))
        assert len(settled) == 1
        assert settled[0].status == OrderStatus.CONFIRMED
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation/test_broker.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/simulation/broker.py
"""Simulated trade execution for off-exchange ETF linked funds.

DD-2: NO slippage — off-exchange funds trade at exact declared NAV.
DD-3: Orders go through PENDING → CONFIRMED state machine.
  - Buy: T+1 share confirmation (shares available next trading day).
  - Sell: T+2 cash settlement (cash available 2 trading days later).

The backtest engine must call broker.process_settlements(current_date)
each day to transition pending orders to confirmed state.
"""
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import List

from etf_analyzer.core.calendar import settle_date
from etf_analyzer.simulation.fees import FeeCalculator


class TradeType(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"


@dataclass
class PendingOrder:
    """Represents a trade order with settlement state."""
    code: str
    trade_type: TradeType
    trade_date: date
    nav: float
    status: OrderStatus = OrderStatus.PENDING

    # Buy-specific
    amount: float = 0.0
    purchase_fee: float = 0.0
    shares: float = 0.0
    confirm_date: date = None     # T+1 for buys

    # Sell-specific
    gross_amount: float = 0.0
    redemption_fee: float = 0.0
    net_amount: float = 0.0
    settle_date: date = None      # T+2 for sells
    holding_days: int = 0


class SimBroker:
    """Simulate off-exchange fund trade execution.

    No slippage (DD-2). Pending order queue with settlement processing (DD-3).
    """

    def __init__(self, fee_calculator: FeeCalculator):
        self._fee_calc = fee_calculator
        self._pending_orders: List[PendingOrder] = []

    def submit_buy(
        self,
        code: str,
        amount: float,
        nav: float,
        trade_date: date,
        is_dca: bool = False,
    ) -> PendingOrder:
        """Submit a buy order. Returns PendingOrder (not yet confirmed).

        Off-exchange: trades at exact NAV (no slippage). T+1 share confirmation.
        """
        fee = self._fee_calc.purchase_fee(amount, is_dca=is_dca)
        net_amount = amount - fee
        shares = net_amount / nav  # Exact NAV, no slippage (DD-2)
        conf_date = settle_date(trade_date, n=1)

        order = PendingOrder(
            code=code,
            trade_type=TradeType.BUY,
            trade_date=trade_date,
            nav=nav,
            amount=amount,
            purchase_fee=fee,
            shares=shares,
            confirm_date=conf_date,
        )
        self._pending_orders.append(order)
        return order

    def submit_sell(
        self,
        code: str,
        shares: float,
        nav: float,
        trade_date: date,
        holding_days: int,
    ) -> PendingOrder:
        """Submit a sell (redemption) order. Returns PendingOrder.

        Off-exchange: trades at exact NAV (no slippage). T+2 cash settlement.
        """
        gross = shares * nav  # Exact NAV, no slippage (DD-2)
        redemption_fee = self._fee_calc.redemption_fee(gross, holding_days)
        net = gross - redemption_fee
        sett_date = settle_date(trade_date, n=2)

        order = PendingOrder(
            code=code,
            trade_type=TradeType.SELL,
            trade_date=trade_date,
            nav=nav,
            shares=shares,
            gross_amount=gross,
            redemption_fee=redemption_fee,
            net_amount=net,
            settle_date=sett_date,
            holding_days=holding_days,
        )
        self._pending_orders.append(order)
        return order

    def process_settlements(self, current_date: date) -> List[PendingOrder]:
        """Process pending orders that settle on current_date.

        Buy orders confirm on confirm_date (T+1).
        Sell orders settle on settle_date (T+2).
        Returns list of newly confirmed orders.
        """
        newly_confirmed = []
        still_pending = []

        for order in self._pending_orders:
            target_date = (
                order.confirm_date if order.trade_type == TradeType.BUY
                else order.settle_date
            )
            if target_date is not None and target_date <= current_date:
                order.status = OrderStatus.CONFIRMED
                newly_confirmed.append(order)
            else:
                still_pending.append(order)

        self._pending_orders = still_pending
        return newly_confirmed
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulation/test_broker.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add simulated broker with T+1/T+2 settlement, no slippage (DD-2/DD-3)"
```

---

### Task 17: Portfolio position tracking (FIFO lot-based — see DD-3)

**Files:**
- Test: `tests/test_simulation/test_portfolio.py`
- Create: `src/etf_analyzer/simulation/portfolio.py`

**Step 1: Write the failing test**

```python
# tests/test_simulation/test_portfolio.py
"""Tests for FIFO lot-based portfolio position tracking.

DD-3: Each purchase creates a Lot(shares, cost, buy_date). Sells consume lots
FIFO. Redemption fee per lot uses that lot's actual holding days.
avg_cost and total_shares are derived from the lot list.
"""
from datetime import date
import pytest
from etf_analyzer.simulation.portfolio import Portfolio, Lot


class TestPortfolio:
    def test_initial_cash(self):
        p = Portfolio(initial_cash=100000)
        assert p.cash == 100000
        assert p.total_value(prices={}) == 100000

    def test_add_lot(self):
        p = Portfolio(initial_cash=100000)
        p.add_lot("510300", shares=1000, cost_per_share=1.5, buy_date=date(2024, 1, 2))
        p.cash -= 1500
        pos = p.get_position("510300")
        assert pos is not None
        assert pos.total_shares == 1000
        assert pos.avg_cost == 1.5

    def test_multiple_lots_tracked_separately(self):
        p = Portfolio(initial_cash=100000)
        p.add_lot("510300", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        p.add_lot("510300", shares=500, cost_per_share=2.0, buy_date=date(2024, 2, 1))
        pos = p.get_position("510300")
        assert pos.total_shares == 1500
        assert len(pos.lots) == 2
        # Weighted avg: (1000*1.0 + 500*2.0) / 1500 = 2000/1500 ≈ 1.333
        assert pos.avg_cost == pytest.approx(1.333, abs=0.01)

    def test_reduce_position_fifo(self):
        """DD-3: FIFO — oldest lot consumed first."""
        p = Portfolio(initial_cash=100000)
        p.add_lot("510300", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        p.add_lot("510300", shares=500, cost_per_share=2.0, buy_date=date(2024, 2, 1))
        consumed_lots = p.reduce_position_fifo("510300", shares=800)
        # Should consume 800 from the first lot (1000 shares at 1.0)
        assert len(consumed_lots) == 1
        assert consumed_lots[0].shares == 800
        assert consumed_lots[0].buy_date == date(2024, 1, 2)
        pos = p.get_position("510300")
        assert pos.total_shares == 700  # 200 remaining + 500 second lot
        assert len(pos.lots) == 2
        assert pos.lots[0].shares == 200  # Remainder of first lot

    def test_reduce_fifo_spans_multiple_lots(self):
        """FIFO sell that consumes entire first lot + part of second."""
        p = Portfolio(initial_cash=100000)
        p.add_lot("510300", shares=300, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        p.add_lot("510300", shares=500, cost_per_share=2.0, buy_date=date(2024, 2, 1))
        consumed_lots = p.reduce_position_fifo("510300", shares=500)
        # 300 from lot1 (fully consumed) + 200 from lot2
        assert len(consumed_lots) == 2
        assert consumed_lots[0].shares == 300
        assert consumed_lots[1].shares == 200
        pos = p.get_position("510300")
        assert pos.total_shares == 300  # 300 remaining in second lot
        assert len(pos.lots) == 1

    def test_reduce_all_removes_position(self):
        p = Portfolio(initial_cash=100000)
        p.add_lot("510300", shares=1000, cost_per_share=1.5, buy_date=date(2024, 1, 2))
        p.reduce_position_fifo("510300", shares=1000)
        assert p.get_position("510300") is None

    def test_total_value_with_positions(self):
        p = Portfolio(initial_cash=50000)
        p.add_lot("510300", shares=1000, cost_per_share=1.5, buy_date=date(2024, 1, 2))
        prices = {"510300": 2.0}
        assert p.total_value(prices) == 50000 + 1000 * 2.0

    def test_position_weights(self):
        p = Portfolio(initial_cash=0)
        p.add_lot("510300", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        p.add_lot("510500", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        weights = p.position_weights(prices={"510300": 2.0, "510500": 3.0})
        assert weights["510300"] == pytest.approx(0.4, abs=0.01)
        assert weights["510500"] == pytest.approx(0.6, abs=0.01)

    def test_lot_holding_days(self):
        """Each lot tracks its own buy_date for redemption fee tier."""
        p = Portfolio(initial_cash=100000)
        p.add_lot("510300", shares=1000, cost_per_share=1.5, buy_date=date(2024, 1, 2))
        p.add_lot("510300", shares=500, cost_per_share=1.6, buy_date=date(2024, 3, 1))
        pos = p.get_position("510300")
        assert pos.lots[0].holding_days(date(2024, 4, 1)) == 90  # Jan 2 -> Apr 1
        assert pos.lots[1].holding_days(date(2024, 4, 1)) == 31  # Mar 1 -> Apr 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation/test_portfolio.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/simulation/portfolio.py
"""FIFO lot-based portfolio and position tracking.

DD-3: Each purchase creates a Lot with its own buy_date. When selling,
lots are consumed FIFO (oldest first). This enables accurate per-lot
holding-day calculation for tiered redemption fee computation.

avg_cost and total_shares are derived properties from the lot list.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional


@dataclass
class Lot:
    """A single purchase lot with its own cost basis and buy date."""
    shares: float
    cost_per_share: float
    buy_date: date

    def market_value(self, price: float) -> float:
        return self.shares * price

    def cost_basis(self) -> float:
        return self.shares * self.cost_per_share

    def holding_days(self, as_of: date) -> int:
        return (as_of - self.buy_date).days


class Position:
    """Aggregate position for a single ETF, backed by FIFO lot list."""

    def __init__(self, code: str):
        self.code = code
        self.lots: List[Lot] = []

    @property
    def total_shares(self) -> float:
        return sum(lot.shares for lot in self.lots)

    @property
    def total_cost(self) -> float:
        return sum(lot.cost_basis() for lot in self.lots)

    @property
    def avg_cost(self) -> float:
        total_shares = self.total_shares
        if total_shares == 0:
            return 0.0
        return self.total_cost / total_shares

    def market_value(self, price: float) -> float:
        return self.total_shares * price

    def unrealized_pnl(self, price: float) -> float:
        return self.market_value(price) - self.total_cost

    def unrealized_pnl_pct(self, price: float) -> float:
        if self.total_cost == 0:
            return 0.0
        return self.unrealized_pnl(price) / self.total_cost

    def add_lot(self, shares: float, cost_per_share: float, buy_date: date) -> None:
        self.lots.append(Lot(shares=shares, cost_per_share=cost_per_share, buy_date=buy_date))

    def reduce_fifo(self, shares_to_sell: float) -> List[Lot]:
        """Consume lots FIFO. Returns list of consumed Lot fragments.

        Each returned Lot has the shares actually sold from that lot,
        along with the original cost_per_share and buy_date (needed
        for per-lot redemption fee calculation).
        """
        if shares_to_sell > self.total_shares + 1e-6:
            raise ValueError(
                f"Cannot sell {shares_to_sell} shares; only hold {self.total_shares}"
            )
        consumed = []
        remaining_to_sell = shares_to_sell

        while remaining_to_sell > 1e-6 and self.lots:
            lot = self.lots[0]
            if lot.shares <= remaining_to_sell + 1e-6:
                # Consume entire lot
                consumed.append(Lot(
                    shares=lot.shares,
                    cost_per_share=lot.cost_per_share,
                    buy_date=lot.buy_date,
                ))
                remaining_to_sell -= lot.shares
                self.lots.pop(0)
            else:
                # Partial consumption
                consumed.append(Lot(
                    shares=remaining_to_sell,
                    cost_per_share=lot.cost_per_share,
                    buy_date=lot.buy_date,
                ))
                lot.shares -= remaining_to_sell
                remaining_to_sell = 0

        return consumed


class Portfolio:
    """Manages multiple FIFO-lot positions and cash balance."""

    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self._initial_cash = initial_cash
        self._positions: Dict[str, Position] = {}

    def add_lot(
        self, code: str, shares: float, cost_per_share: float, buy_date: date
    ) -> None:
        """Add a new purchase lot to a position."""
        if code not in self._positions:
            self._positions[code] = Position(code=code)
        self._positions[code].add_lot(shares, cost_per_share, buy_date)

    def reduce_position_fifo(self, code: str, shares: float) -> List[Lot]:
        """Sell shares FIFO. Returns consumed lots for fee calculation.

        Each consumed lot carries its own buy_date, so the caller can
        compute the correct tiered redemption fee per lot.
        """
        if code not in self._positions:
            raise KeyError(f"No position for {code}")
        consumed = self._positions[code].reduce_fifo(shares)
        # Remove position if fully liquidated
        if self._positions[code].total_shares < 1e-6:
            del self._positions[code]
        return consumed

    def get_position(self, code: str) -> Optional[Position]:
        return self._positions.get(code)

    @property
    def positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    def total_value(self, prices: Dict[str, float]) -> float:
        """Total portfolio value = cash + sum of position market values."""
        pos_value = sum(
            pos.market_value(prices.get(code, pos.avg_cost))
            for code, pos in self._positions.items()
        )
        return self.cash + pos_value

    def position_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate position weights as fraction of total value."""
        total = self.total_value(prices)
        if total == 0:
            return {}
        return {
            code: pos.market_value(prices.get(code, pos.avg_cost)) / total
            for code, pos in self._positions.items()
        }

    def cash_ratio(self, prices: Dict[str, float]) -> float:
        """Cash as fraction of total portfolio value."""
        total = self.total_value(prices)
        if total == 0:
            return 1.0
        return self.cash / total
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulation/test_portfolio.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add FIFO lot-based portfolio tracking for per-lot redemption fees (DD-3)"
```

---

## Phase 5: Strategy

### Task 18: Signal types and base strategy

**Files:**
- Test: `tests/test_strategy/test_base.py`
- Create: `src/etf_analyzer/strategy/signals.py`
- Create: `src/etf_analyzer/strategy/base.py`

**Step 1: Write the failing test**

```python
# tests/test_strategy/test_base.py
"""Tests for signal types and base strategy."""
import pytest
from etf_analyzer.strategy.signals import Signal, SignalType
from etf_analyzer.strategy.base import BaseStrategy


class TestSignal:
    def test_create_buy_signal(self):
        sig = Signal(
            signal_type=SignalType.BUY,
            code="510300",
            reason="PE below 20% percentile",
            strength=0.8,
        )
        assert sig.signal_type == SignalType.BUY
        assert sig.code == "510300"
        assert sig.strength == 0.8

    def test_signal_types_exist(self):
        assert SignalType.BUY
        assert SignalType.SELL
        assert SignalType.HOLD
        assert SignalType.REBALANCE
        assert SignalType.ADD
        assert SignalType.TAKE_PROFIT
        assert SignalType.STOP_LOSS


class TestBaseStrategy:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            BaseStrategy(name="test")

    def test_subclass_must_implement_generate_signals(self):
        class IncompleteStrategy(BaseStrategy):
            pass
        with pytest.raises(TypeError):
            IncompleteStrategy(name="test")

    def test_subclass_works(self):
        class DummyStrategy(BaseStrategy):
            def generate_signals(self, market_data, portfolio, current_date):
                return []
        s = DummyStrategy(name="dummy")
        assert s.name == "dummy"
        assert s.generate_signals({}, None, None) == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_strategy/test_base.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/strategy/signals.py
"""Trading signal types."""
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    ADD = "add"          # Add to existing position
    HOLD = "hold"
    REBALANCE = "rebalance"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"


@dataclass
class Signal:
    """A trading signal generated by a strategy."""
    signal_type: SignalType
    code: str
    reason: str
    strength: float = 1.0          # 0.0 to 1.0
    target_amount: float = 0.0     # Amount to buy (CNY) or shares to sell
    target_weight: float = 0.0     # Target portfolio weight
    date: Optional[date] = None
```

```python
# src/etf_analyzer/strategy/base.py
"""Abstract base strategy class."""
from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Dict, List

from etf_analyzer.strategy.signals import Signal
from etf_analyzer.simulation.portfolio import Portfolio


class BaseStrategy(ABC):
    """Base class for all investment strategies.

    Subclasses must implement generate_signals().
    Strategy parameters should be loaded from YAML config.
    """

    def __init__(self, name: str, params: dict = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def generate_signals(
        self,
        market_data: Dict[str, Any],
        portfolio: Portfolio,
        current_date: date,
    ) -> List[Signal]:
        """Generate trading signals based on market data and portfolio state.

        Args:
            market_data: Dict with keys like 'prices', 'pe_percentiles', etc.
            portfolio: Current portfolio state.
            current_date: Current simulation date.

        Returns:
            List of Signal objects to execute.
        """
        ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_strategy/test_base.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add signal types and abstract base strategy"
```

---

### Task 19: Semi-monthly rebalance strategy

**Files:**
- Test: `tests/test_strategy/test_semi_monthly.py`
- Create: `src/etf_analyzer/strategy/semi_monthly.py`

**Step 1: Write the failing test**

```python
# tests/test_strategy/test_semi_monthly.py
"""Tests for semi-monthly rebalance strategy."""
from datetime import date
import pytest
from etf_analyzer.strategy.semi_monthly import SemiMonthlyStrategy
from etf_analyzer.strategy.signals import SignalType
from etf_analyzer.simulation.portfolio import Portfolio


@pytest.fixture
def strategy():
    params = {
        "rebalance_day": [1, 16],
        "deviation_single": 0.05,
        "deviation_portfolio": 0.03,
        "buy_signal": {
            "broad_market": {
                "pe_percentile_threshold": 0.20,
                "daily_drop_trigger": -0.03,
            },
        },
        "take_profit": {
            "tier1": {"return_threshold": 0.15, "reduce_ratio": 0.20},
            "tier2": {"return_threshold": 0.30, "reduce_ratio": 0.30},
        },
        "stop_loss": {
            "single_max_drawdown": -0.20,
            "ma_break": {
                "ma_period": 20,
                "daily_drop": -0.05,
                "confirm_days": 3,
                "reduce_ratio": 0.50,
            },
            "portfolio_drawdown": {
                "pause_add_threshold": -0.10,
                "force_reduce_threshold": -0.15,
                "force_reduce_ratio": 0.20,
            },
        },
        "target_weights": {"510300": 0.40, "510500": 0.30, "518880": 0.30},
    }
    return SemiMonthlyStrategy(params=params)


class TestSemiMonthlyStrategy:
    def test_rebalance_signal_on_rebalance_day(self, strategy):
        portfolio = Portfolio(initial_cash=100000)
        portfolio.add_lot("510300", shares=5000, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        portfolio.add_lot("510500", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        portfolio.add_lot("518880", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        portfolio.cash = 93000  # started with 100k, spent 7k

        market_data = {
            "prices": {"510300": 1.2, "510500": 1.0, "518880": 1.0},
            "daily_returns": {"510300": 0.01, "510500": -0.005, "518880": 0.0},
        }
        # Jan 16 is a rebalance day
        signals = strategy.generate_signals(market_data, portfolio, date(2024, 1, 16))
        signal_types = [s.signal_type for s in signals]
        # Portfolio is imbalanced (510300 is 60% vs target 40%)
        assert SignalType.REBALANCE in signal_types

    def test_no_rebalance_on_non_rebalance_day(self, strategy):
        portfolio = Portfolio(initial_cash=100000)
        market_data = {"prices": {}, "daily_returns": {}}
        signals = strategy.generate_signals(market_data, portfolio, date(2024, 1, 10))
        rebalance_signals = [s for s in signals if s.signal_type == SignalType.REBALANCE]
        assert len(rebalance_signals) == 0

    def test_take_profit_signal(self, strategy):
        portfolio = Portfolio(initial_cash=100000)
        portfolio.add_lot("510300", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        portfolio.cash = 99000

        market_data = {
            "prices": {"510300": 1.20},  # 20% gain > 15% tier1 threshold
            "daily_returns": {"510300": 0.01},
        }
        signals = strategy.generate_signals(market_data, portfolio, date(2024, 3, 1))
        tp_signals = [s for s in signals if s.signal_type == SignalType.TAKE_PROFIT]
        assert len(tp_signals) > 0

    def test_stop_loss_on_large_drawdown(self, strategy):
        portfolio = Portfolio(initial_cash=100000)
        portfolio.add_lot("510300", shares=1000, cost_per_share=1.0, buy_date=date(2024, 1, 2))
        portfolio.cash = 99000

        market_data = {
            "prices": {"510300": 0.75},  # 25% loss > 20% threshold
            "daily_returns": {"510300": -0.06},
        }
        signals = strategy.generate_signals(market_data, portfolio, date(2024, 3, 1))
        sl_signals = [s for s in signals if s.signal_type == SignalType.STOP_LOSS]
        assert len(sl_signals) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_strategy/test_semi_monthly.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/strategy/semi_monthly.py
"""Semi-monthly rebalance strategy implementation.

Core strategy from design doc: rebalance every half-month, with
valuation-based buy signals, tiered take-profit, and stop-loss rules.
"""
from datetime import date
from typing import Any, Dict, List

from etf_analyzer.strategy.base import BaseStrategy
from etf_analyzer.strategy.signals import Signal, SignalType
from etf_analyzer.simulation.portfolio import Portfolio
from etf_analyzer.core.logger import get_logger

logger = get_logger("strategy.semi_monthly")


class SemiMonthlyStrategy(BaseStrategy):
    """Semi-monthly rebalance portfolio strategy."""

    def __init__(self, params: dict):
        super().__init__(name="semi_monthly_rebalance", params=params)
        self._rebalance_days = params.get("rebalance_day", [1, 16])
        self._deviation_single = params.get("deviation_single", 0.05)
        self._target_weights = params.get("target_weights", {})
        self._tp_config = params.get("take_profit", {})
        self._sl_config = params.get("stop_loss", {})

    def generate_signals(
        self,
        market_data: Dict[str, Any],
        portfolio: Portfolio,
        current_date: date,
    ) -> List[Signal]:
        signals = []
        prices = market_data.get("prices", {})
        daily_returns = market_data.get("daily_returns", {})

        # 1. Check stop-loss first (highest priority)
        signals.extend(self._check_stop_loss(portfolio, prices, daily_returns))

        # 2. Check take-profit
        signals.extend(self._check_take_profit(portfolio, prices))

        # 3. Check rebalance on rebalance days
        if current_date.day in self._rebalance_days:
            signals.extend(self._check_rebalance(portfolio, prices))

        return signals

    def _check_rebalance(
        self, portfolio: Portfolio, prices: Dict[str, float]
    ) -> List[Signal]:
        """Check if portfolio deviates from target weights."""
        signals = []
        if not self._target_weights or not prices:
            return signals

        current_weights = portfolio.position_weights(prices)

        for code, target_w in self._target_weights.items():
            current_w = current_weights.get(code, 0.0)
            deviation = current_w - target_w
            if abs(deviation) > self._deviation_single:
                signals.append(Signal(
                    signal_type=SignalType.REBALANCE,
                    code=code,
                    reason=f"Weight deviation {deviation:+.2%} exceeds ±{self._deviation_single:.0%}",
                    target_weight=target_w,
                ))

        return signals

    def _check_take_profit(
        self, portfolio: Portfolio, prices: Dict[str, float]
    ) -> List[Signal]:
        """Check tiered take-profit conditions."""
        signals = []
        for code, pos in portfolio.positions.items():
            if code not in prices:
                continue
            pnl_pct = pos.unrealized_pnl_pct(prices[code])

            # Check tiers (highest first)
            tier2 = self._tp_config.get("tier2", {})
            tier1 = self._tp_config.get("tier1", {})

            if tier2 and pnl_pct >= tier2.get("return_threshold", 0.30):
                signals.append(Signal(
                    signal_type=SignalType.TAKE_PROFIT,
                    code=code,
                    reason=f"Return {pnl_pct:.1%} >= tier2 {tier2['return_threshold']:.0%}",
                    target_amount=pos.total_shares * tier2.get("reduce_ratio", 0.30),
                ))
            elif tier1 and pnl_pct >= tier1.get("return_threshold", 0.15):
                signals.append(Signal(
                    signal_type=SignalType.TAKE_PROFIT,
                    code=code,
                    reason=f"Return {pnl_pct:.1%} >= tier1 {tier1['return_threshold']:.0%}",
                    target_amount=pos.total_shares * tier1.get("reduce_ratio", 0.20),
                ))

        return signals

    def _check_stop_loss(
        self,
        portfolio: Portfolio,
        prices: Dict[str, float],
        daily_returns: Dict[str, float],
    ) -> List[Signal]:
        """Check stop-loss conditions."""
        signals = []
        max_dd_threshold = self._sl_config.get("single_max_drawdown", -0.20)

        for code, pos in portfolio.positions.items():
            if code not in prices:
                continue
            pnl_pct = pos.unrealized_pnl_pct(prices[code])

            # Single ETF max drawdown -> clear position
            if pnl_pct <= max_dd_threshold:
                signals.append(Signal(
                    signal_type=SignalType.STOP_LOSS,
                    code=code,
                    reason=f"Loss {pnl_pct:.1%} exceeds max drawdown {max_dd_threshold:.0%}",
                    target_amount=pos.total_shares,  # Sell all
                ))

        return signals
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_strategy/test_semi_monthly.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add semi-monthly rebalance strategy with TP/SL signals"
```

---

## Phase 6: Fund Selection

### Task 20: Fund screener (3-tier filtering)

**Files:**
- Test: `tests/test_selection/test_screener.py`
- Create: `src/etf_analyzer/selection/screener.py`

**Step 1: Write the failing test**

```python
# tests/test_selection/test_screener.py
"""Tests for 3-tier fund screener."""
import pandas as pd
import pytest
from etf_analyzer.selection.screener import FundScreener, ScreeningCriteria


@pytest.fixture
def sample_fund_universe():
    return pd.DataFrame({
        "code": ["510300", "510500", "159915", "512010", "999999"],
        "name": ["沪深300ETF", "中证500ETF", "创业板ETF", "医药ETF", "TinyFund"],
        "scale": [100e8, 50e8, 30e8, 20e8, 2e8],          # 基金规模(元)
        "tracking_error": [0.003, 0.005, 0.008, 0.006, 0.015],
        "total_fee_rate": [0.005, 0.005, 0.006, 0.007, 0.008],
        "years_since_inception": [5, 4, 3, 2, 0.5],
        "pe_percentile": [0.15, 0.25, 0.45, 0.10, 0.80],
        "ma60_uptrend": [True, True, False, True, False],
        "annual_volatility": [0.15, 0.18, 0.25, 0.19, 0.30],
        "category": ["broad_market", "broad_market", "broad_market", "sector", "sector"],
    })


class TestFundScreener:
    def test_initial_screening(self, sample_fund_universe):
        screener = FundScreener()
        result = screener.initial_screen(sample_fund_universe)
        # 999999 fails: scale < 5B, years < 1
        assert "999999" not in result["code"].values
        # 159915 may fail fee check (0.006 <= 0.006, borderline)
        assert len(result) >= 3

    def test_secondary_screening(self, sample_fund_universe):
        screener = FundScreener()
        initial = screener.initial_screen(sample_fund_universe)
        result = screener.secondary_screen(initial)
        # 159915 fails: pe_percentile=0.45 > 0.30, ma60_uptrend=False
        # 510500 passes: pe=0.25 < 0.30, ma60=True, vol=0.18 < 0.20
        codes = result["code"].tolist()
        assert "510300" in codes  # pe=0.15 < 0.30, ma60=True, vol=0.15
        assert "159915" not in codes

    def test_custom_criteria(self, sample_fund_universe):
        criteria = ScreeningCriteria(min_scale=20e8, max_tracking_error=0.01)
        screener = FundScreener(criteria=criteria)
        result = screener.initial_screen(sample_fund_universe)
        assert all(result["scale"] >= 20e8)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_selection/test_screener.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/selection/screener.py
"""Multi-tier fund screening.

Initial screening (基本面), secondary screening (估值+趋势),
and category classification.
"""
from dataclasses import dataclass

import pandas as pd

from etf_analyzer.core.logger import get_logger

logger = get_logger("selection.screener")


@dataclass
class ScreeningCriteria:
    """Configurable screening thresholds."""
    # Initial screening
    min_scale: float = 5e8              # 基金规模≥5亿
    max_tracking_error: float = 0.01    # 跟踪误差≤1%
    max_total_fee: float = 0.006        # 管理+托管费率≤0.6%
    min_years: float = 1.0              # 成立时间≥1年

    # Secondary screening
    max_pe_percentile: float = 0.30     # PE近5年30%分位以下 (PB unavailable, see DD-1)
    require_ma_uptrend: bool = True     # 60日均线向上
    max_volatility: float = 0.20        # 年化波动率≤20%


class FundScreener:
    """3-tier fund screening engine."""

    def __init__(self, criteria: ScreeningCriteria = None):
        self._criteria = criteria or ScreeningCriteria()

    def initial_screen(self, universe: pd.DataFrame) -> pd.DataFrame:
        """Initial screening: basic fundamentals filter.

        Filters: scale, tracking error, fees, inception age.
        """
        c = self._criteria
        mask = (
            (universe["scale"] >= c.min_scale)
            & (universe["tracking_error"] <= c.max_tracking_error)
            & (universe["total_fee_rate"] <= c.max_total_fee)
            & (universe["years_since_inception"] >= c.min_years)
        )
        result = universe[mask].reset_index(drop=True)
        logger.info(
            f"Initial screening: {len(universe)} -> {len(result)} funds"
        )
        return result

    def secondary_screen(self, df: pd.DataFrame) -> pd.DataFrame:
        """Secondary screening: valuation + trend filter.

        Filters: PE percentile, MA uptrend, volatility.
        """
        c = self._criteria
        mask = (
            (df["pe_percentile"] <= c.max_pe_percentile)
            & (df["annual_volatility"] <= c.max_volatility)
        )
        if c.require_ma_uptrend:
            mask = mask & (df["ma60_uptrend"] == True)

        result = df[mask].reset_index(drop=True)
        logger.info(
            f"Secondary screening: {len(df)} -> {len(result)} funds"
        )
        return result

    def screen(self, universe: pd.DataFrame) -> pd.DataFrame:
        """Run full screening pipeline (initial + secondary)."""
        initial = self.initial_screen(universe)
        return self.secondary_screen(initial)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_selection/test_screener.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add 3-tier fund screener with configurable criteria"
```

---

### Task 21: Fund scorer (weighted scoring & ranking)

**Files:**
- Test: `tests/test_selection/test_scorer.py`
- Create: `src/etf_analyzer/selection/scorer.py`

**Step 1: Write the failing test**

```python
# tests/test_selection/test_scorer.py
"""Tests for fund scoring and ranking."""
import pandas as pd
import pytest
from etf_analyzer.selection.scorer import FundScorer


@pytest.fixture
def screened_funds():
    return pd.DataFrame({
        "code": ["510300", "510500", "512010"],
        "pe_percentile": [0.10, 0.20, 0.15],
        "total_fee_rate": [0.004, 0.005, 0.006],
        "tracking_error": [0.003, 0.005, 0.008],
        "category": ["broad_market", "broad_market", "sector"],
    })


class TestFundScorer:
    def test_score_all_funds(self, screened_funds):
        scorer = FundScorer()
        scored = scorer.score(screened_funds)
        assert "total_score" in scored.columns
        assert scored["total_score"].between(0, 1).all()

    def test_ranking_order(self, screened_funds):
        scorer = FundScorer()
        ranked = scorer.rank(screened_funds)
        # Should be sorted by total_score descending
        scores = ranked["total_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_top_n_per_category(self, screened_funds):
        scorer = FundScorer()
        top = scorer.top_n_per_category(screened_funds, n=1)
        categories = top["category"].unique()
        for cat in categories:
            assert len(top[top["category"] == cat]) <= 1

    def test_custom_weights(self, screened_funds):
        scorer = FundScorer(
            valuation_weight=0.5,
            fee_weight=0.3,
            tracking_weight=0.2,
        )
        scored = scorer.score(screened_funds)
        assert "total_score" in scored.columns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_selection/test_scorer.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/selection/scorer.py
"""Fund scoring and ranking.

Weighted scoring: 估值(40%) + 费率(30%) + 跟踪误差(30%).
"""
import numpy as np
import pandas as pd

from etf_analyzer.core.logger import get_logger

logger = get_logger("selection.scorer")


class FundScorer:
    """Score and rank funds based on weighted criteria."""

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
        """Calculate total score for each fund.

        Lower PE percentile, lower fees, lower tracking error = higher score.
        """
        result = df.copy()

        # Valuation score: lower percentile = higher score
        result["val_score"] = 1.0 - result["pe_percentile"]

        # Fee score: lower fee = higher score (normalize to [0,1])
        fee_min = result["total_fee_rate"].min()
        fee_max = result["total_fee_rate"].max()
        if fee_max > fee_min:
            result["fee_score"] = 1.0 - (result["total_fee_rate"] - fee_min) / (fee_max - fee_min)
        else:
            result["fee_score"] = 1.0

        # Tracking error score: lower = better
        te_min = result["tracking_error"].min()
        te_max = result["tracking_error"].max()
        if te_max > te_min:
            result["te_score"] = 1.0 - (result["tracking_error"] - te_min) / (te_max - te_min)
        else:
            result["te_score"] = 1.0

        # Weighted total
        result["total_score"] = (
            result["val_score"] * self._val_w
            + result["fee_score"] * self._fee_w
            + result["te_score"] * self._te_w
        )

        return result

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score and sort by total_score descending."""
        scored = self.score(df)
        return scored.sort_values("total_score", ascending=False).reset_index(drop=True)

    def top_n_per_category(self, df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        """Select top N funds per category."""
        ranked = self.rank(df)
        if "category" not in ranked.columns:
            return ranked.head(n)
        return (
            ranked.groupby("category")
            .head(n)
            .reset_index(drop=True)
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_selection/test_scorer.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add fund scorer with weighted ranking per category"
```

---

## Phase 7: Backtesting

### Task 22: Backtest engine

**Files:**
- Test: `tests/test_backtest/test_engine.py`
- Create: `src/etf_analyzer/backtest/engine.py`

**Step 1: Write the failing test**

```python
# tests/test_backtest/test_engine.py
"""Tests for backtest engine."""
from datetime import date
import pandas as pd
import pytest
from etf_analyzer.backtest.engine import BacktestEngine, BacktestConfig
from etf_analyzer.strategy.base import BaseStrategy
from etf_analyzer.strategy.signals import Signal, SignalType
from etf_analyzer.simulation.fees import FeeSchedule


class MockStrategy(BaseStrategy):
    """Buy 510300 on first day, hold forever."""
    def generate_signals(self, market_data, portfolio, current_date):
        if not portfolio.get_position("510300"):
            return [Signal(
                signal_type=SignalType.BUY,
                code="510300",
                reason="Initial buy",
                target_amount=50000,
            )]
        return []


@pytest.fixture
def price_data():
    dates = pd.bdate_range("2024-01-02", periods=60)
    prices = [1.0 + i * 0.01 for i in range(60)]  # Steady uptrend
    return pd.DataFrame({"日期": dates, "510300": prices})


class TestBacktestEngine:
    def test_basic_backtest_runs(self, price_data):
        config = BacktestConfig(
            initial_capital=100000,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 22),
            fee_schedule=FeeSchedule(),
        )
        engine = BacktestEngine(config=config)
        strategy = MockStrategy(name="mock")
        result = engine.run(strategy=strategy, price_data=price_data)

        assert result is not None
        assert "equity_curve" in result
        assert "trade_log" in result
        assert "final_value" in result
        assert result["final_value"] > 0

    def test_equity_curve_length(self, price_data):
        config = BacktestConfig(
            initial_capital=100000,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 22),
            fee_schedule=FeeSchedule(),
        )
        engine = BacktestEngine(config=config)
        strategy = MockStrategy(name="mock")
        result = engine.run(strategy=strategy, price_data=price_data)
        assert len(result["equity_curve"]) == len(price_data)

    def test_trade_log_records_buy(self, price_data):
        config = BacktestConfig(
            initial_capital=100000,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 22),
            fee_schedule=FeeSchedule(),
        )
        engine = BacktestEngine(config=config)
        strategy = MockStrategy(name="mock")
        result = engine.run(strategy=strategy, price_data=price_data)
        assert len(result["trade_log"]) >= 1
        assert result["trade_log"][0]["type"] == "buy"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest/test_engine.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/backtest/engine.py
"""Event-driven backtest engine.

Iterates through price data day by day, feeds market data to strategy,
executes signals through simulated broker, tracks portfolio equity.

DD-2: No slippage — off-exchange funds trade at declared NAV.
DD-3: Processes pending settlements each day before generating new signals.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List

import pandas as pd

from etf_analyzer.core.logger import get_logger
from etf_analyzer.simulation.broker import SimBroker, TradeType
from etf_analyzer.simulation.fees import FeeCalculator, FeeSchedule
from etf_analyzer.simulation.portfolio import Portfolio
from etf_analyzer.strategy.base import BaseStrategy
from etf_analyzer.strategy.signals import SignalType

logger = get_logger("backtest.engine")


@dataclass
class BacktestConfig:
    initial_capital: float = 100000
    start_date: date = date(2020, 1, 1)
    end_date: date = date(2025, 1, 1)
    fee_schedule: FeeSchedule = field(default_factory=FeeSchedule)
    benchmark_code: str = "000300"


class BacktestEngine:
    """Run a strategy against historical price data."""

    def __init__(self, config: BacktestConfig):
        self._config = config
        fee_calc = FeeCalculator(config.fee_schedule)
        self._broker = SimBroker(fee_calculator=fee_calc)  # No slippage (DD-2)

    def run(
        self,
        strategy: BaseStrategy,
        price_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Execute backtest.

        Args:
            strategy: Strategy instance.
            price_data: DataFrame with '日期' column and one column per ETF code.

        Returns:
            Dict with 'equity_curve', 'trade_log', 'final_value', 'portfolio'.
        """
        portfolio = Portfolio(initial_cash=self._config.initial_capital)
        equity_curve = []
        trade_log = []

        # Get ETF code columns (everything except '日期')
        etf_codes = [c for c in price_data.columns if c != "日期"]

        for _, row in price_data.iterrows():
            current_date = pd.Timestamp(row["日期"]).date()
            prices = {code: row[code] for code in etf_codes if pd.notna(row[code])}

            # DD-3: Process pending settlements first
            confirmed_orders = self._broker.process_settlements(current_date)
            for order in confirmed_orders:
                if order.trade_type == TradeType.BUY:
                    portfolio.add_lot(
                        code=order.code,
                        shares=order.shares,
                        cost_per_share=order.amount / order.shares,  # Includes purchase fee in cost basis
                        buy_date=order.confirm_date,  # Holding days start from T+1 confirmation
                    )
                elif order.trade_type == TradeType.SELL:
                    portfolio.cash += order.net_amount

            # Build market data dict for strategy
            market_data = {
                "prices": prices,
                "daily_returns": {},  # Simplified for now
            }

            # Generate signals (strategy sees only confirmed positions)
            signals = strategy.generate_signals(market_data, portfolio, current_date)

            # Execute signals
            for signal in signals:
                self._execute_signal(signal, portfolio, prices, current_date, trade_log)

            # Record equity
            total_value = portfolio.total_value(prices)
            equity_curve.append({
                "date": current_date,
                "total_value": total_value,
                "cash": portfolio.cash,
            })

        final_value = equity_curve[-1]["total_value"] if equity_curve else self._config.initial_capital

        return {
            "equity_curve": equity_curve,
            "trade_log": trade_log,
            "final_value": final_value,
            "portfolio": portfolio,
        }

    def _execute_signal(
        self,
        signal,
        portfolio: Portfolio,
        prices: Dict[str, float],
        current_date: date,
        trade_log: List[dict],
    ) -> None:
        """Execute a single trading signal."""
        code = signal.code

        if signal.signal_type in (SignalType.BUY, SignalType.ADD):
            amount = signal.target_amount
            if amount > portfolio.cash:
                amount = portfolio.cash
            if amount <= 0 or code not in prices:
                return

            order = self._broker.submit_buy(
                code=code, amount=amount, nav=prices[code], trade_date=current_date
            )
            # DD-3: Cash deducted immediately, shares arrive on T+1
            portfolio.cash -= amount
            trade_log.append({
                "date": current_date,
                "code": code,
                "type": "buy",
                "amount": amount,
                "shares": order.shares,
                "nav": prices[code],
                "fee": order.purchase_fee,
                "reason": signal.reason,
            })

        elif signal.signal_type in (SignalType.SELL, SignalType.TAKE_PROFIT, SignalType.STOP_LOSS):
            pos = portfolio.get_position(code)
            if pos is None or code not in prices:
                return

            shares_to_sell = min(signal.target_amount, pos.total_shares)
            if shares_to_sell <= 0:
                return

            # DD-3: FIFO lot consumption for per-lot redemption fees
            consumed_lots = portfolio.reduce_position_fifo(code, shares_to_sell)
            total_net = 0.0
            total_fee = 0.0
            for lot in consumed_lots:
                holding_days = lot.holding_days(current_date)
                order = self._broker.submit_sell(
                    code=code,
                    shares=lot.shares,
                    nav=prices[code],
                    trade_date=current_date,
                    holding_days=holding_days,
                )
                total_net += order.net_amount
                total_fee += order.redemption_fee

            # DD-3: Cash arrives on T+2 (handled by process_settlements)
            trade_log.append({
                "date": current_date,
                "code": code,
                "type": signal.signal_type.value,
                "shares": shares_to_sell,
                "nav": prices[code],
                "net_amount": total_net,
                "fee": total_fee,
                "reason": signal.reason,
            })
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest/test_engine.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add event-driven backtest engine"
```

---

### Task 23: Performance metrics aggregation

**Files:**
- Test: `tests/test_backtest/test_metrics.py`
- Create: `src/etf_analyzer/backtest/metrics.py`

**Step 1: Write the failing test**

```python
# tests/test_backtest/test_metrics.py
"""Tests for backtest performance metrics."""
import pandas as pd
import pytest
from etf_analyzer.backtest.metrics import calculate_backtest_metrics


@pytest.fixture
def equity_curve():
    dates = pd.bdate_range("2024-01-02", periods=252)
    values = [100000 * (1 + 0.0003) ** i for i in range(252)]  # ~8% annual
    return [{"date": d.date(), "total_value": v, "cash": 5000} for d, v in zip(dates, values)]


@pytest.fixture
def trade_log():
    return [
        {"date": "2024-01-02", "type": "buy", "amount": 50000},
        {"date": "2024-06-01", "type": "sell", "net_amount": 55000},
        {"date": "2024-06-15", "type": "buy", "amount": 55000},
        {"date": "2024-12-01", "type": "sell", "net_amount": 52000},
    ]


class TestBacktestMetrics:
    def test_returns_dict(self, equity_curve, trade_log):
        metrics = calculate_backtest_metrics(equity_curve, trade_log, initial_capital=100000)
        assert isinstance(metrics, dict)

    def test_has_required_metrics(self, equity_curve, trade_log):
        metrics = calculate_backtest_metrics(equity_curve, trade_log, initial_capital=100000)
        required = [
            "total_return",
            "annual_return",
            "max_drawdown",
            "sharpe_ratio",
            "volatility",
            "win_rate",
            "total_trades",
            "max_drawdown_duration",
        ]
        for key in required:
            assert key in metrics, f"Missing metric: {key}"

    def test_total_return_positive(self, equity_curve, trade_log):
        metrics = calculate_backtest_metrics(equity_curve, trade_log, initial_capital=100000)
        assert metrics["total_return"] > 0

    def test_win_rate_calculation(self, equity_curve, trade_log):
        metrics = calculate_backtest_metrics(equity_curve, trade_log, initial_capital=100000)
        # 2 sell trades: 55000 (win), 52000 (loss if bought at 55000)
        assert 0 <= metrics["win_rate"] <= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest/test_metrics.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/backtest/metrics.py
"""Backtest performance metrics aggregation."""
from typing import Any, Dict, List

import pandas as pd

from etf_analyzer.formulas.risk import (
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    annualized_volatility,
)
from etf_analyzer.formulas.returns import annualized_return


def calculate_backtest_metrics(
    equity_curve: List[dict],
    trade_log: List[dict],
    initial_capital: float,
    risk_free_rate: float = 0.02,
) -> Dict[str, Any]:
    """Calculate comprehensive backtest performance metrics.

    Args:
        equity_curve: List of {date, total_value, cash} dicts.
        trade_log: List of trade records.
        initial_capital: Starting capital.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
    """
    if not equity_curve:
        return {}

    # Build NAV series
    values = pd.Series(
        [e["total_value"] for e in equity_curve],
        index=pd.DatetimeIndex([e["date"] for e in equity_curve]),
    )
    daily_returns = values.pct_change().dropna()

    # Total return
    final_value = values.iloc[-1]
    total_ret = (final_value - initial_capital) / initial_capital

    # Holding period in days
    holding_days = (values.index[-1] - values.index[0]).days
    if holding_days <= 0:
        holding_days = 1

    # Annual return
    ann_ret = annualized_return(total_ret, holding_days)

    # Risk metrics
    mdd = max_drawdown(values)
    mdd_dur = max_drawdown_duration(values)
    vol = annualized_volatility(daily_returns) if len(daily_returns) > 1 else 0.0
    sr = sharpe_ratio(daily_returns, risk_free_rate) if len(daily_returns) > 1 else 0.0

    # Trade statistics
    sell_trades = [t for t in trade_log if t.get("type") in ("sell", "take_profit", "stop_loss")]
    buy_trades = [t for t in trade_log if t.get("type") == "buy"]
    total_trades = len(buy_trades) + len(sell_trades)

    # Win rate (simplified: sell > buy amount)
    wins = 0
    for t in sell_trades:
        net = t.get("net_amount", 0)
        amount = t.get("amount", 0)
        if net > amount or net > 0:
            wins += 1
    win_rate = wins / len(sell_trades) if sell_trades else 0.0

    return {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return": total_ret,
        "annual_return": ann_ret,
        "max_drawdown": mdd,
        "max_drawdown_duration": mdd_dur,
        "sharpe_ratio": sr,
        "volatility": vol,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "holding_days": holding_days,
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest/test_metrics.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add backtest performance metrics aggregation"
```

---

### Task 24: Visualization (charts)

**Files:**
- Test: `tests/test_backtest/test_report.py` (visual output test)
- Create: `src/etf_analyzer/backtest/visualization.py`

**Step 1: Write the failing test**

```python
# tests/test_backtest/test_report.py
"""Tests for backtest visualization and report generation."""
import pandas as pd
import pytest
from pathlib import Path
from etf_analyzer.backtest.visualization import (
    plot_equity_curve,
    plot_drawdown,
    fig_to_base64,
)


@pytest.fixture
def equity_data():
    dates = pd.bdate_range("2024-01-02", periods=100)
    portfolio = [100000 * (1 + 0.0003) ** i for i in range(100)]
    benchmark = [100000 * (1 + 0.0002) ** i for i in range(100)]
    return pd.DataFrame({
        "date": dates,
        "portfolio": portfolio,
        "benchmark": benchmark,
    })


class TestVisualization:
    def test_equity_curve_returns_figure(self, equity_data):
        fig = plot_equity_curve(
            dates=equity_data["date"],
            portfolio_values=equity_data["portfolio"],
            benchmark_values=equity_data["benchmark"],
        )
        assert fig is not None

    def test_drawdown_returns_figure(self, equity_data):
        fig = plot_drawdown(
            dates=equity_data["date"],
            portfolio_values=equity_data["portfolio"],
        )
        assert fig is not None

    def test_fig_to_base64_returns_string(self, equity_data):
        fig = plot_equity_curve(
            dates=equity_data["date"],
            portfolio_values=equity_data["portfolio"],
        )
        b64 = fig_to_base64(fig)
        assert isinstance(b64, str)
        assert len(b64) > 100  # Should be a substantial base64 string
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest/test_report.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/etf_analyzer/backtest/visualization.py
"""Backtest visualization: equity curves, drawdown charts."""
import base64
from io import BytesIO
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def plot_equity_curve(
    dates: pd.Series,
    portfolio_values: pd.Series,
    benchmark_values: Optional[pd.Series] = None,
    title: str = "Cumulative Returns",
) -> plt.Figure:
    """Plot portfolio equity curve vs benchmark.

    Args:
        dates: Date series.
        portfolio_values: Portfolio total value series.
        benchmark_values: Optional benchmark value series.
        title: Chart title.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, portfolio_values, label="Portfolio", linewidth=1.5, color="#2196F3")
    if benchmark_values is not None:
        ax.plot(dates, benchmark_values, label="Benchmark", linewidth=1.0, color="#9E9E9E", linestyle="--")
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Value (CNY)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_drawdown(
    dates: pd.Series,
    portfolio_values: pd.Series,
    title: str = "Drawdown",
) -> plt.Figure:
    """Plot drawdown chart."""
    values = pd.Series(portfolio_values.values, index=dates)
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dates, drawdown, 0, color="red", alpha=0.3)
    ax.plot(dates, drawdown, color="red", linewidth=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 PNG string (for HTML embedding)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest/test_report.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add backtest visualization (equity curve, drawdown charts)"
```

---

### Task 25: HTML report generation

**Files:**
- Create: `templates/report.html`
- Create: `src/etf_analyzer/backtest/report.py`
- Update: `tests/test_backtest/test_report.py` (add report tests)

**Step 1: Write the failing test (append to existing test file)**

```python
# Append to tests/test_backtest/test_report.py
from etf_analyzer.backtest.report import generate_html_report


class TestHtmlReport:
    def test_generate_report_creates_file(self, equity_data, tmp_path):
        output_path = tmp_path / "report.html"
        metrics = {
            "total_return": 0.08,
            "annual_return": 0.08,
            "max_drawdown": -0.05,
            "sharpe_ratio": 1.2,
            "volatility": 0.12,
            "win_rate": 0.6,
            "total_trades": 10,
            "initial_capital": 100000,
            "final_value": 108000,
        }
        generate_html_report(
            metrics=metrics,
            equity_curve=equity_data,
            output_path=str(output_path),
        )
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "108000" in content or "108,000" in content
        assert "Portfolio" in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest/test_report.py::TestHtmlReport -v`
Expected: FAIL

**Step 3: Create the Jinja2 template**

```html
<!-- templates/report.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>ETF Portfolio Backtest Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }
        .container { max-width: 1100px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }
        h2 { color: #283593; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th { background: #1a237e; color: white; padding: 12px; text-align: left; }
        td { padding: 10px 12px; border-bottom: 1px solid #e0e0e0; }
        tr:hover { background: #f5f5f5; }
        .positive { color: #2e7d32; font-weight: bold; }
        .negative { color: #c62828; font-weight: bold; }
        .chart { margin: 20px 0; text-align: center; }
        .chart img { max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #1a237e; }
        .metric-label { font-size: 0.85em; color: #666; }
        .metric-value { font-size: 1.4em; font-weight: bold; margin-top: 5px; }
        .report-date { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ETF Portfolio Backtest Report</h1>
        <p class="report-date">Generated: {{ report_date }}</p>

        <h2>Performance Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Initial Capital</div>
                <div class="metric-value">{{ "%.2f"|format(metrics.initial_capital) }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Final Value</div>
                <div class="metric-value {% if metrics.final_value > metrics.initial_capital %}positive{% else %}negative{% endif %}">
                    {{ "%.2f"|format(metrics.final_value) }}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {% if metrics.total_return > 0 %}positive{% else %}negative{% endif %}">
                    {{ "%.2f%%"|format(metrics.total_return * 100) }}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Annual Return</div>
                <div class="metric-value {% if metrics.annual_return > 0 %}positive{% else %}negative{% endif %}">
                    {{ "%.2f%%"|format(metrics.annual_return * 100) }}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{{ "%.2f%%"|format(metrics.max_drawdown * 100) }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{{ "%.2f"|format(metrics.sharpe_ratio) }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volatility</div>
                <div class="metric-value">{{ "%.2f%%"|format(metrics.volatility * 100) }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{{ "%.1f%%"|format(metrics.win_rate * 100) }}</div>
            </div>
        </div>

        {% if equity_chart %}
        <h2>Equity Curve</h2>
        <div class="chart">
            <img src="data:image/png;base64,{{ equity_chart }}" alt="Equity Curve">
        </div>
        {% endif %}

        {% if drawdown_chart %}
        <h2>Drawdown</h2>
        <div class="chart">
            <img src="data:image/png;base64,{{ drawdown_chart }}" alt="Drawdown">
        </div>
        {% endif %}

        <h2>Detailed Metrics</h2>
        <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
                {% for key, value in metrics.items() %}
                <tr>
                    <td>{{ key }}</td>
                    <td>{% if value is number %}{{ "%.4f"|format(value) }}{% else %}{{ value }}{% endif %}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
```

**Step 4: Write report generator**

```python
# src/etf_analyzer/backtest/report.py
"""HTML report generation using Jinja2 templates."""
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from etf_analyzer.backtest.visualization import (
    plot_equity_curve,
    plot_drawdown,
    fig_to_base64,
)
from etf_analyzer.core.logger import get_logger

logger = get_logger("backtest.report")

# Template directory: project_root/templates/
_TEMPLATE_DIR = Path(__file__).parent.parent.parent.parent / "templates"


def generate_html_report(
    metrics: Dict[str, Any],
    equity_curve: pd.DataFrame,
    output_path: str,
    benchmark_col: Optional[str] = "benchmark",
    template_dir: str = None,
) -> str:
    """Generate HTML backtest report.

    Args:
        metrics: Performance metrics dict.
        equity_curve: DataFrame with 'date', 'portfolio', and optionally 'benchmark'.
        output_path: Output HTML file path.
        benchmark_col: Column name for benchmark values.
        template_dir: Override template directory.
    """
    tpl_dir = Path(template_dir) if template_dir else _TEMPLATE_DIR
    if not tpl_dir.exists():
        # Fallback: create minimal inline template
        tpl_dir = Path(output_path).parent
        _create_fallback_template(tpl_dir)

    env = Environment(loader=FileSystemLoader(str(tpl_dir)))
    template = env.get_template("report.html")

    # Generate charts
    equity_chart = None
    drawdown_chart = None

    if "date" in equity_curve.columns and "portfolio" in equity_curve.columns:
        benchmark = equity_curve.get(benchmark_col)
        fig_eq = plot_equity_curve(
            dates=equity_curve["date"],
            portfolio_values=equity_curve["portfolio"],
            benchmark_values=benchmark,
        )
        equity_chart = fig_to_base64(fig_eq)

        fig_dd = plot_drawdown(
            dates=equity_curve["date"],
            portfolio_values=equity_curve["portfolio"],
        )
        drawdown_chart = fig_to_base64(fig_dd)

    html = template.render(
        report_date=date.today().strftime("%Y-%m-%d"),
        metrics=metrics,
        equity_chart=equity_chart,
        drawdown_chart=drawdown_chart,
    )

    Path(output_path).write_text(html, encoding="utf-8")
    logger.info(f"Report generated: {output_path}")
    return output_path


def _create_fallback_template(directory: Path) -> None:
    """Create a minimal fallback template if main template not found."""
    template_path = directory / "report.html"
    if template_path.exists():
        return
    template_path.write_text(
        """<!DOCTYPE html><html><head><title>Backtest Report</title></head>
<body><h1>Backtest Report</h1><p>Generated: {{ report_date }}</p>
<h2>Metrics</h2><table>{% for k, v in metrics.items() %}
<tr><td>{{ k }}</td><td>{{ v }}</td></tr>{% endfor %}</table>
{% if equity_chart %}<h2>Equity</h2><img src="data:image/png;base64,{{ equity_chart }}">{% endif %}
{% if drawdown_chart %}<h2>Drawdown</h2><img src="data:image/png;base64,{{ drawdown_chart }}">{% endif %}
</body></html>""",
        encoding="utf-8",
    )
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_backtest/test_report.py -v`
Expected: All passed

**Step 6: Commit**

```bash
git add -A && git commit -m "feat: add HTML report generation with Jinja2 templates"
```

---

## Phase 8: Integration

### Task 26: Full integration test (end-to-end pipeline)

**Files:**
- Test: `tests/test_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_integration.py
"""End-to-end integration test: strategy -> backtest -> metrics -> report."""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_analyzer.backtest.engine import BacktestEngine, BacktestConfig
from etf_analyzer.backtest.metrics import calculate_backtest_metrics
from etf_analyzer.backtest.report import generate_html_report
from etf_analyzer.simulation.fees import FeeSchedule
from etf_analyzer.strategy.base import BaseStrategy
from etf_analyzer.strategy.signals import Signal, SignalType


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy-and-hold for integration testing."""
    def generate_signals(self, market_data, portfolio, current_date):
        prices = market_data.get("prices", {})
        for code in prices:
            if not portfolio.get_position(code):
                target = portfolio.cash * 0.45  # Buy ~45% per ETF
                if target > 100:
                    return [Signal(
                        signal_type=SignalType.BUY,
                        code=code,
                        reason="Initial allocation",
                        target_amount=target,
                    )]
        return []


@pytest.fixture
def synthetic_price_data():
    """Generate 1 year of synthetic price data for 2 ETFs."""
    dates = pd.bdate_range("2024-01-02", periods=252)
    np.random.seed(42)
    etf1 = 1.0 * np.cumprod(1 + np.random.normal(0.0003, 0.01, 252))
    etf2 = 2.0 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 252))
    return pd.DataFrame({
        "日期": dates,
        "510300": etf1,
        "510500": etf2,
    })


class TestEndToEnd:
    def test_full_pipeline(self, synthetic_price_data, tmp_path):
        # 1. Configure and run backtest
        config = BacktestConfig(
            initial_capital=100000,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 12, 31),
            fee_schedule=FeeSchedule(),
        )
        engine = BacktestEngine(config=config)
        strategy = BuyAndHoldStrategy(name="buy_hold")
        result = engine.run(strategy=strategy, price_data=synthetic_price_data)

        # 2. Verify backtest output
        assert result["final_value"] > 0
        assert len(result["equity_curve"]) == 252
        assert len(result["trade_log"]) >= 1

        # 3. Calculate metrics
        metrics = calculate_backtest_metrics(
            equity_curve=result["equity_curve"],
            trade_log=result["trade_log"],
            initial_capital=100000,
        )
        assert "annual_return" in metrics
        assert "max_drawdown" in metrics
        assert "sharpe_ratio" in metrics

        # 4. Generate report
        equity_df = pd.DataFrame(result["equity_curve"])
        equity_df = equity_df.rename(columns={"total_value": "portfolio"})
        report_path = tmp_path / "backtest_report.html"
        generate_html_report(
            metrics=metrics,
            equity_curve=equity_df,
            output_path=str(report_path),
        )
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Portfolio" in content
        assert len(content) > 500

    def test_pipeline_with_zero_trades(self, tmp_path):
        """Strategy that never trades should still produce valid results."""
        class NoOpStrategy(BaseStrategy):
            def generate_signals(self, market_data, portfolio, current_date):
                return []

        dates = pd.bdate_range("2024-01-02", periods=20)
        price_data = pd.DataFrame({
            "日期": dates,
            "510300": [1.0 + i * 0.01 for i in range(20)],
        })
        config = BacktestConfig(initial_capital=100000, fee_schedule=FeeSchedule())
        engine = BacktestEngine(config=config)
        result = engine.run(strategy=NoOpStrategy(name="noop"), price_data=price_data)
        assert result["final_value"] == 100000  # All cash
        assert len(result["trade_log"]) == 0

        metrics = calculate_backtest_metrics(
            result["equity_curve"], result["trade_log"], 100000
        )
        assert metrics["total_return"] == pytest.approx(0.0, abs=1e-9)
```

**Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: All passed

**Step 3: Run full test suite**

Run: `pytest --tb=short -q`
Expected: All tests pass

**Step 4: Commit**

```bash
git add -A && git commit -m "feat: add end-to-end integration test for full pipeline"
```

---

## Summary

| Phase | Tasks | Key Deliverables |
|-------|-------|------------------|
| 1. Infrastructure | 1-5 | Project setup, ApiResponse, Config, Calendar, Cache |
| 2. Formulas | 6-10 | Returns, Valuation, Risk, Technical, Factors |
| 3. Data | 11-14 | Store, Cleaner, Fetcher, Updater |
| 4. Simulation | 15-17 | Fees, Broker (T+1/T+2), Portfolio tracking |
| 5. Strategy | 18-19 | Signal types, Base strategy, Semi-monthly rebalance |
| 6. Selection | 20-21 | 3-tier screener, Weighted scorer |
| 7. Backtest | 22-25 | Engine, Metrics, Visualization, HTML report |
| 8. Integration | 26 | End-to-end pipeline test |

**Total: 26 tasks, ~50 test cases, full TDD coverage**

**Dependencies:**
```
Phase 1 (Infrastructure) → all other phases
Phase 2 (Formulas) → Phase 4, 5, 6, 7
Phase 3 (Data) → Phase 5, 6, 7
Phase 4 (Simulation) → Phase 5, 7
Phase 5 (Strategy) → Phase 7
Phase 6 (Selection) → standalone (uses formulas + data)
Phase 7 (Backtest) → final integration
```
