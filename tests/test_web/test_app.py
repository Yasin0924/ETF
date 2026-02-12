"""Tests for Flask web application."""

import pytest
import etf_analyzer.web.app as web_app
from etf_analyzer.web.app import create_app


@pytest.fixture
def client():
    app = create_app(config={"TESTING": True})
    with app.test_client() as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status_code"] == 0
        assert data["data"]["status"] == "ok"

    def test_name_map_skips_remote_lookup_by_default(self, monkeypatch):
        monkeypatch.setattr(web_app, "_enable_spot_name_lookup", False)

        def _boom():
            raise AssertionError("should not call data components")

        monkeypatch.setattr(web_app, "_get_data_components", _boom)
        assert web_app._name_map_from_spot() == {}

    def test_system_status_returns_ok(self, client):
        resp = client.get("/api/system/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status_code"] == 0
        assert "data_source" in data["data"]
        assert "weekly_full_check" in data["data"]

    def test_system_refresh_returns_ok(self, client):
        resp = client.post("/api/system/refresh", json={"scope": "all"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status_code"] == 0
        assert data["data"]["refresh_requested"] is True


class TestBacktestEndpoint:
    def test_run_backtest_default(self, client):
        resp = client.post("/api/backtest/run", json={})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status_code"] == 0
        assert data["data"]["metrics"]["initial_capital"] == 100000
        assert "data_source" in data["data"]["config"]

    def test_run_backtest_custom_capital(self, client):
        resp = client.post("/api/backtest/run", json={"initial_capital": 200000})
        data = resp.get_json()
        assert data["data"]["metrics"]["initial_capital"] == 200000

    def test_run_backtest_invalid_capital(self, client):
        resp = client.post("/api/backtest/run", json={"initial_capital": -100})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status_code"] == 2

    def test_validate_backtest(self, client):
        resp = client.post(
            "/api/backtest/validate",
            json={
                "train_ratio": 0.7,
                "initial_capital": 120000,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status_code"] == 0
        assert data["data"]["train_ratio"] == 0.7
        assert "train_metrics" in data["data"]
        assert "validation_metrics" in data["data"]

    def test_sensitivity_backtest(self, client):
        resp = client.post(
            "/api/backtest/sensitivity",
            json={
                "param": "deviation_single",
                "values": [0.03, 0.04, 0.05],
                "initial_capital": 100000,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status_code"] == 0
        assert data["data"]["param"] == "deviation_single"
        assert len(data["data"]["points"]) == 3


class TestEtfDataEndpoint:
    def test_get_etf_data(self, client):
        resp = client.get("/api/data/etf/510300")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status_code"] == 0
        assert data["data"]["code"] == "510300"
        assert "source" in data["data"]

    def test_get_etf_data_with_params(self, client):
        resp = client.get("/api/data/etf/159915?start=2023-01-01&end=2024-01-01")
        data = resp.get_json()
        assert data["data"]["start"] == "2023-01-01"
        assert data["data"]["end"] == "2024-01-01"


class TestScreeningEndpoint:
    def test_screen_funds(self, client):
        resp = client.post(
            "/api/selection/screen", json={"min_size": 1e9, "categories": ["宽基"]}
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status_code"] == 0
        assert data["data"]["criteria"]["min_size"] == 1e9

    def test_screen_funds_default(self, client):
        resp = client.post("/api/selection/screen", json={})
        data = resp.get_json()
        assert data["data"]["criteria"]["min_size"] == 1e8

    def test_screen_funds_empty_returns_hint_not_error(self, client):
        resp = client.post(
            "/api/selection/screen",
            json={
                "min_size": 0,
                "max_fee": 0.005,
                "categories": ["宽基", "红利", "黄金", "行业成长"],
                "pe_percentile_max": 0.1,
                "pb_percentile_max": 0.1,
                "roe_min": 0.2,
                "tracking_error_max": 0.001,
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status_code"] == 0
        assert data["data"]["total"] == 0
        assert "hint" in data["data"]
        assert "suggested_criteria" in data["data"]["diagnostics"]


class TestDcaEndpoint:
    def test_run_dca(self, client):
        resp = client.post(
            "/api/dca/run",
            json={
                "code": "510300",
                "amount": 1000,
                "mode": "fixed_amount",
                "interval_days": 30,
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status_code"] == 0
        assert data["data"]["code"] == "510300"
        assert "data_source" in data["data"]

    def test_run_dca_invalid_amount(self, client):
        resp = client.post("/api/dca/run", json={"amount": 0})
        assert resp.status_code == 400


class TestStrategyParamsEndpoint:
    def test_get_params(self, client):
        resp = client.get("/api/strategy/params")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status_code"] == 0

    def test_put_params(self, client):
        resp = client.put("/api/strategy/params", json={"custom_key": "custom_value"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status_code"] == 0
        assert data["data"]["custom_key"] == "custom_value"

    def test_put_empty_body_returns_error(self, client):
        resp = client.put(
            "/api/strategy/params",
            data="",
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_strategy_signals(self, client):
        resp = client.post("/api/strategy/signals", json={"code": "510300"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status_code"] == 0
        assert data["data"]["code"] == "510300"


class TestPageRoutes:
    def test_dashboard_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_backtest_page_returns_200(self, client):
        resp = client.get("/backtest")
        assert resp.status_code == 200

    def test_backtest_page_has_validation_and_sensitivity_modules(self, client):
        resp = client.get("/backtest")
        body = resp.get_data(as_text=True)
        assert "样本外验证" in body
        assert "参数敏感性分析" in body

    def test_dashboard_page_has_risk_status_section(self, client):
        resp = client.get("/")
        body = resp.get_data(as_text=True)
        assert "风控状态" in body

    def test_dashboard_page_has_system_status_section(self, client):
        resp = client.get("/")
        body = resp.get_data(as_text=True)
        assert "数据服务状态" in body
        assert "刷新数据" in body

    def test_screening_page_returns_200(self, client):
        resp = client.get("/screening")
        assert resp.status_code == 200

    def test_screening_page_has_add_to_pool_action(self, client):
        resp = client.get("/screening")
        body = resp.get_data(as_text=True)
        assert "加入策略池" in body

    def test_screening_page_has_sort_and_export_entry(self, client):
        resp = client.get("/screening")
        body = resp.get_data(as_text=True)
        assert "点击表头可排序" in body
        assert "导出筛选结果" in body

    def test_strategy_page_returns_200(self, client):
        resp = client.get("/strategy")
        assert resp.status_code == 200

    def test_strategy_page_has_candidate_pool_section(self, client):
        resp = client.get("/strategy")
        body = resp.get_data(as_text=True)
        assert "候选基金池" in body

    def test_dca_page_returns_200(self, client):
        resp = client.get("/dca")
        assert resp.status_code == 200

    def test_dca_page_uses_chinese_nav_label(self, client):
        resp = client.get("/dca")
        body = resp.get_data(as_text=True)
        assert "单位净值" in body

    def test_data_page_returns_200(self, client):
        resp = client.get("/data")
        assert resp.status_code == 200

    def test_data_page_uses_chinese_nav_label(self, client):
        resp = client.get("/data")
        body = resp.get_data(as_text=True)
        assert "单位净值" in body

    def test_backtest_page_uses_chinese_nav_label(self, client):
        resp = client.get("/backtest")
        body = resp.get_data(as_text=True)
        assert "单位净值" in body

    def test_strategy_page_has_chinese_json_hint(self, client):
        resp = client.get("/strategy")
        body = resp.get_data(as_text=True)
        assert "对象/数组文本格式" in body

    def test_strategy_page_has_validation_hint(self, client):
        resp = client.get("/strategy")
        body = resp.get_data(as_text=True)
        assert "参数校验提示" in body

    def test_backtest_page_has_export_entry(self, client):
        resp = client.get("/backtest")
        body = resp.get_data(as_text=True)
        assert "导出回测结果" in body

    def test_backtest_page_has_operation_status_area(self, client):
        resp = client.get("/backtest")
        body = resp.get_data(as_text=True)
        assert 'id="op-status"' in body

    def test_dca_page_has_dividend_mode_entry(self, client):
        resp = client.get("/dca")
        body = resp.get_data(as_text=True)
        assert "分红方式" in body

    def test_dca_page_has_operation_status_area(self, client):
        resp = client.get("/dca")
        body = resp.get_data(as_text=True)
        assert 'id="op-status"' in body

    def test_data_page_has_operation_status_area(self, client):
        resp = client.get("/data")
        body = resp.get_data(as_text=True)
        assert 'id="op-status"' in body

    def test_screening_page_has_operation_status_area(self, client):
        resp = client.get("/screening")
        body = resp.get_data(as_text=True)
        assert 'id="op-status"' in body

    def test_strategy_page_has_operation_status_area(self, client):
        resp = client.get("/strategy")
        body = resp.get_data(as_text=True)
        assert 'id="op-status"' in body
