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
        df = pd.DataFrame(
            {"日期": ["2024-01-02"], "pe": [12.5], "dividend_yield": [2.5]}
        )
        store.save_index_data("000300", df)
        loaded = store.load_index_data("000300")
        assert loaded is not None
        assert len(loaded) == 1
