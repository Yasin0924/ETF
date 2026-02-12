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
