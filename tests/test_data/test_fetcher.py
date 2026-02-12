"""Tests for akshare data fetcher wrapper."""

import pandas as pd
from unittest.mock import patch
from etf_analyzer.data.fetcher import EtfDataFetcher
from etf_analyzer.core.response import StatusCode


class TestEtfDataFetcher:
    def test_fetch_etf_history_uses_fallback_source(self):
        with patch("etf_analyzer.data.fetcher.ak") as mock_ak:
            mock_ak.fund_etf_hist_em.side_effect = Exception("primary failed")
            fallback_df = pd.DataFrame(
                {
                    "日期": ["2024-01-02"],
                    "开盘": [1.0],
                    "收盘": [1.1],
                    "最高": [1.2],
                    "最低": [0.9],
                    "成交量": [10000],
                    "成交额": [11000.0],
                    "涨跌幅": [1.0],
                }
            )
            mock_ak.fund_etf_hist_sina.return_value = fallback_df
            fetcher = EtfDataFetcher(retry_count=1)
            resp = fetcher.fetch_etf_history("510300", "20240101", "20240201")
            assert resp.ok
            assert len(resp.data) == 1

    def test_fetch_etf_history_hits_cache(self, tmp_path):
        with patch("etf_analyzer.data.fetcher.ak") as mock_ak:
            mock_df = pd.DataFrame(
                {
                    "日期": ["2024-01-02"],
                    "开盘": [1.0],
                    "收盘": [1.1],
                    "最高": [1.2],
                    "最低": [0.9],
                    "成交量": [10000],
                    "成交额": [11000.0],
                    "涨跌幅": [1.0],
                }
            )
            mock_ak.fund_etf_hist_em.return_value = mock_df
            fetcher = EtfDataFetcher(retry_count=1, cache_dir=str(tmp_path / "cache"))
            first = fetcher.fetch_etf_history("510300", "20240101", "20240201")
            second = fetcher.fetch_etf_history("510300", "20240101", "20240201")
            assert first.ok and second.ok
            assert mock_ak.fund_etf_hist_em.call_count == 1

    def test_fetch_etf_hist_success(self):
        mock_df = pd.DataFrame(
            {
                "日期": ["2024-01-02"],
                "开盘": [1.0],
                "收盘": [1.1],
                "最高": [1.2],
                "最低": [0.9],
                "成交量": [10000],
                "成交额": [11000.0],
                "涨跌幅": [1.0],
            }
        )
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
        mock_df = pd.DataFrame(
            {
                "项目": ["管理费率", "托管费率", "销售服务费率"],
                "数据": ["0.50%", "0.10%", "0.00%"],
            }
        )
        with patch("etf_analyzer.data.fetcher.ak") as mock_ak:
            mock_ak.fund_fee_em.return_value = mock_df
            fetcher = EtfDataFetcher()
            resp = fetcher.fetch_fund_fee("510300", indicator="运作费用")
            assert resp.ok
            assert len(resp.data) > 0

    def test_fetch_index_valuation_success(self):
        mock_df = pd.DataFrame(
            {
                "日期": ["2024-01-02"],
                "市盈率1": [12.5],
                "股息率1": [2.5],
            }
        )
        with patch("etf_analyzer.data.fetcher.ak") as mock_ak:
            mock_ak.stock_zh_index_value_csindex.return_value = mock_df
            fetcher = EtfDataFetcher()
            resp = fetcher.fetch_index_valuation("000300")
            assert resp.ok
