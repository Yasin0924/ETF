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
        df = pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "收盘": [1.0, np.nan, 1.2],
                "成交量": [100, 200, 300],
            }
        )
        cleaned = clean_nav_data(df)
        assert len(cleaned) == 2
        assert cleaned["收盘"].isna().sum() == 0

    def test_removes_negative_nav(self):
        df = pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03"],
                "收盘": [1.0, -0.5],
                "成交量": [100, 200],
            }
        )
        cleaned = clean_nav_data(df)
        assert len(cleaned) == 1

    def test_sorts_by_date(self):
        df = pd.DataFrame(
            {
                "日期": ["2024-01-04", "2024-01-02", "2024-01-03"],
                "收盘": [1.2, 1.0, 1.1],
                "成交量": [100, 200, 300],
            }
        )
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
        df = pd.DataFrame(
            {
                "日期": ["2024-01-02"],
                "开盘": [1.0],
                "收盘": [1.1],
                "最高": [0.9],
                "最低": [1.0],
                "成交量": [100],
            }
        )
        errors = validate_ohlcv(df)
        assert any("最高" in e or "high" in e.lower() for e in errors)


class TestFillMissingDates:
    def test_fills_gap(self):
        df = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2024-01-02", "2024-01-04"]),
                "收盘": [1.0, 1.2],
            }
        )
        filled = fill_missing_dates(df)
        assert len(filled) == 3
