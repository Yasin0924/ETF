"""Tests for incremental data updater."""

from datetime import date
import pandas as pd
from unittest.mock import MagicMock
from etf_analyzer.data.updater import DataUpdater
from etf_analyzer.data.store import EtfDataStore
from etf_analyzer.core.response import ApiResponse


class TestDataUpdater:
    def test_full_update_new_etf(self, tmp_data_dir):
        store = EtfDataStore(data_dir=str(tmp_data_dir))
        mock_fetcher = MagicMock()
        mock_df = pd.DataFrame(
            {
                "日期": pd.date_range("2024-01-02", periods=5).strftime("%Y-%m-%d"),
                "收盘": [1.0, 1.1, 1.2, 1.1, 1.3],
                "开盘": [1.0, 1.0, 1.1, 1.2, 1.1],
                "最高": [1.1, 1.2, 1.3, 1.2, 1.4],
                "最低": [0.9, 1.0, 1.1, 1.0, 1.1],
                "成交量": [100] * 5,
                "成交额": [100.0] * 5,
                "涨跌幅": [0, 10, 9, -8, 18],
            }
        )
        mock_fetcher.fetch_etf_history.return_value = ApiResponse.success(data=mock_df)
        updater = DataUpdater(fetcher=mock_fetcher, store=store)
        result = updater.update_etf("510300")
        assert result.ok
        stored = store.load_etf_data("510300")
        assert stored is not None
        assert len(stored) == 5

    def test_incremental_update_existing_etf(self, tmp_data_dir):
        store = EtfDataStore(data_dir=str(tmp_data_dir))
        old_df = pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "收盘": [1.0, 1.1, 1.2],
                "开盘": [1.0, 1.0, 1.1],
                "最高": [1.1, 1.2, 1.3],
                "最低": [0.9, 1.0, 1.1],
                "成交量": [100, 100, 100],
                "成交额": [100.0, 100.0, 100.0],
                "涨跌幅": [0, 10, 9],
            }
        )
        store.save_etf_data("510300", old_df)
        mock_fetcher = MagicMock()
        new_df = pd.DataFrame(
            {
                "日期": ["2024-01-05", "2024-01-08"],
                "收盘": [1.3, 1.25],
                "开盘": [1.2, 1.3],
                "最高": [1.4, 1.35],
                "最低": [1.2, 1.2],
                "成交量": [200, 150],
                "成交额": [200.0, 150.0],
                "涨跌幅": [8, -4],
            }
        )
        mock_fetcher.fetch_etf_history.return_value = ApiResponse.success(data=new_df)
        updater = DataUpdater(fetcher=mock_fetcher, store=store)
        result = updater.update_etf("510300", incremental=True)
        assert result.ok
        stored = store.load_etf_data("510300")
        if stored is None:
            raise AssertionError("Expected stored dataframe")
        assert len(stored) == 5

    def test_weekly_full_check_policy(self, tmp_data_dir):
        store = EtfDataStore(data_dir=str(tmp_data_dir))
        mock_fetcher = MagicMock()
        mock_df = pd.DataFrame(
            {
                "日期": ["2024-01-02"],
                "收盘": [1.0],
                "开盘": [1.0],
                "最高": [1.1],
                "最低": [0.9],
                "成交量": [100],
                "成交额": [100.0],
                "涨跌幅": [0],
            }
        )
        mock_fetcher.fetch_etf_history.return_value = ApiResponse.success(data=mock_df)
        updater = DataUpdater(fetcher=mock_fetcher, store=store)
        result = updater.update_batch_with_policy(
            ["510300"],
            on_date=date(2026, 2, 8),
            update_config={
                "daily_increment": True,
                "weekly_full_check": True,
                "full_check_day": "Sunday",
            },
        )
        assert result.ok
        assert result.data["mode"] == "full"
