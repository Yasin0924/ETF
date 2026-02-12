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
        config_file.write_text(
            yaml.dump({"backtest": {"initial_capital": 100000, "benchmark": "000300"}})
        )
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
