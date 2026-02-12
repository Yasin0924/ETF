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
