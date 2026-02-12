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
