"""Flask web application for ETF Portfolio Analysis."""

from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from random import Random
from typing import Any, Dict, cast

import pandas as pd

from flask import Flask, jsonify, render_template, request
from jinja2 import TemplateNotFound

from etf_analyzer.core.config import load_config
from etf_analyzer.core.logger import get_logger
from etf_analyzer.core.response import ApiResponse
from etf_analyzer.backtest.engine import BacktestConfig, BacktestEngine
from etf_analyzer.backtest.metrics import calculate_backtest_metrics
from etf_analyzer.strategy.semi_monthly import SemiMonthlyStrategy
from etf_analyzer.simulation.dca import DCASimulator
from etf_analyzer.simulation.fees import FeeCalculator, FeeSchedule
from etf_analyzer.simulation.portfolio import Portfolio
from etf_analyzer.data.fetcher import EtfDataFetcher
from etf_analyzer.data.store import EtfDataStore
from etf_analyzer.data.updater import DataUpdater

logger = get_logger("web.app")

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"

_strategy_params_cache: Dict[str, Any] = {}
_last_refresh_at: str | None = None
_data_store: EtfDataStore | None = None
_data_fetcher: EtfDataFetcher | None = None
_data_updater: DataUpdater | None = None
_allow_synthetic_fallback: bool = False
_enable_spot_name_lookup: bool = False


def _get_data_components() -> tuple[EtfDataStore, EtfDataFetcher, DataUpdater]:
    global _data_store, _data_fetcher, _data_updater
    if (
        _data_store is not None
        and _data_fetcher is not None
        and _data_updater is not None
    ):
        return _data_store, _data_fetcher, _data_updater

    settings_path = _CONFIG_DIR / "settings.yaml"
    settings = load_config(str(settings_path)) if settings_path.exists() else {}
    data_dir = str((settings.get("system") or {}).get("data_dir", "data"))
    cache_dir = str(Path(data_dir) / "cache")

    _data_store = EtfDataStore(data_dir=data_dir)
    _data_fetcher = EtfDataFetcher(cache_dir=cache_dir)
    _data_updater = DataUpdater(fetcher=_data_fetcher, store=_data_store)
    return _data_store, _data_fetcher, _data_updater


def _get_categories() -> dict[str, list[str]]:
    ds_cfg_path = _CONFIG_DIR / "data_sources.yaml"
    ds_cfg = load_config(str(ds_cfg_path)) if ds_cfg_path.exists() else {}
    return ds_cfg.get("etf_categories", {}) if isinstance(ds_cfg, dict) else {}


def _category_for_code(code: str) -> str:
    cats = _get_categories()
    label_map = {
        "broad_market": "宽基",
        "sector": "行业成长",
        "dividend": "红利",
        "gold": "黄金",
    }
    for k, codes in cats.items():
        if code in (codes or []):
            return label_map.get(k, k)
    return "其他"


def _all_etf_codes() -> list[str]:
    cats = _get_categories()
    seq: list[str] = []
    for codes in cats.values():
        for c in codes or []:
            if c not in seq:
                seq.append(c)
    return seq or ["510300", "510500", "518880"]


def _name_map_from_spot() -> dict[str, str]:
    if _allow_synthetic_fallback or not _enable_spot_name_lookup:
        return {}
    _, fetcher, _ = _get_data_components()
    resp = fetcher.fetch_etf_spot()
    if not resp.ok or not isinstance(resp.data, pd.DataFrame):
        return {}
    df = resp.data
    code_col = None
    name_col = None
    for c in ["代码", "基金代码", "symbol", "code"]:
        if c in df.columns:
            code_col = c
            break
    for c in ["名称", "基金简称", "name"]:
        if c in df.columns:
            name_col = c
            break
    if not code_col or not name_col:
        return {}
    return {
        str(row[code_col]): str(row[name_col])
        for _, row in df.iterrows()
        if pd.notna(row[code_col]) and pd.notna(row[name_col])
    }


def _to_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def _filter_range(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if "日期" not in df.columns:
        return pd.DataFrame()
    dt = _to_date_series(cast(pd.Series, df["日期"]))
    mask = (dt >= start_date) & (dt <= end_date)
    out = df.loc[mask].copy()
    if out.empty:
        return out
    out["日期"] = _to_date_series(cast(pd.Series, out["日期"]))
    return out.sort_values("日期").reset_index(drop=True)


def _ensure_real_etf_data(
    code: str, start_date: date, end_date: date
) -> tuple[pd.DataFrame, str]:
    store, _, updater = _get_data_components()
    source = "local_csv"
    existing = store.load_etf_data(code)

    if _allow_synthetic_fallback:
        if existing is None or existing.empty:
            return pd.DataFrame(), "synthetic"
        return _filter_range(existing, start_date, end_date), source

    if existing is None or existing.empty:
        resp = updater.update_etf(
            code=code,
            incremental=False,
            end_date=end_date.strftime("%Y%m%d"),
        )
        if not resp.ok:
            return pd.DataFrame(), "unavailable"
        existing = store.load_etf_data(code)
        source = "live_akshare"
    else:
        clipped = _filter_range(existing, start_date, end_date)
        if clipped.empty:
            resp = updater.update_etf(
                code=code,
                incremental=True,
                end_date=end_date.strftime("%Y%m%d"),
            )
            if resp.ok:
                source = "live_akshare"
                existing = store.load_etf_data(code)

    if existing is None or existing.empty:
        return pd.DataFrame(), "unavailable"
    return _filter_range(existing, start_date, end_date), source


def _require_real_data(df: pd.DataFrame, context: str) -> None:
    if df.empty and not _allow_synthetic_fallback:
        raise RuntimeError(f"{context} 无可用真实数据，请先执行数据更新")


def _screen_real_funds(criteria: dict) -> tuple[list[dict], dict]:
    name_map = _name_map_from_spot()
    end_date = date.today()
    start_date = end_date - timedelta(days=400)
    benchmark_df, _ = _ensure_real_etf_data("510300", start_date, end_date)
    bench_ret = pd.Series(dtype=float)
    if not benchmark_df.empty and "收盘" in benchmark_df.columns:
        bench_ret = (
            pd.to_numeric(benchmark_df["收盘"], errors="coerce").pct_change().dropna()
        )

    results: list[dict] = []
    universe: list[dict] = []
    for code in _all_etf_codes():
        df, _ = _ensure_real_etf_data(code, start_date, end_date)
        if df.empty or "收盘" not in df.columns or len(df) < 40:
            continue
        close = pd.to_numeric(df["收盘"], errors="coerce").dropna()
        if close.empty:
            continue
        returns = close.pct_change().dropna()
        if returns.empty:
            continue

        latest = float(close.iloc[-1])
        pe_pct = float((close < latest).mean())
        pb_pct = pe_pct
        annual_ret = float((latest / float(close.iloc[0])) - 1)
        roe = max(0.0, min(0.3, annual_ret))
        volume = 0.0
        if "成交量" in df.columns:
            volume = float(
                pd.to_numeric(df["成交量"], errors="coerce").fillna(0).iloc[-1]
            )
        size = max(volume * latest, 1e8)
        tracking_error = float(returns.std() * (252**0.5))
        if not bench_ret.empty:
            join = pd.concat(
                [returns.reset_index(drop=True), bench_ret.reset_index(drop=True)],
                axis=1,
            ).dropna()
            if not join.empty:
                tracking_error = float(
                    (join.iloc[:, 0] - join.iloc[:, 1]).std() * (252**0.5)
                )

        fee = 0.005
        item = {
            "code": code,
            "name": name_map.get(code, f"ETF {code}"),
            "category": _category_for_code(code),
            "size": size,
            "fee": fee,
            "pe_percentile": pe_pct,
            "pb_percentile": pb_pct,
            "roe": roe,
            "tracking_error": tracking_error,
            "return_1y": annual_ret,
            "source": "real",
        }

        universe.append(item)

        if item["size"] < criteria["min_size"]:
            continue
        if item["fee"] > criteria["max_fee"]:
            continue
        if criteria["categories"] and item["category"] not in criteria["categories"]:
            continue
        if item["pe_percentile"] > criteria["pe_percentile_max"]:
            continue
        if item["pb_percentile"] > criteria["pb_percentile_max"]:
            continue
        if item["roe"] < criteria["roe_min"]:
            continue
        if item["tracking_error"] > criteria["tracking_error_max"]:
            continue
        results.append(item)

    diagnostics: dict[str, Any] = {
        "universe_total": len(universe),
        "matched_total": len(results),
        "suggested_criteria": {},
    }
    if universe:
        pe_vals = sorted(float(x["pe_percentile"]) for x in universe)
        pb_vals = sorted(float(x["pb_percentile"]) for x in universe)
        te_vals = sorted(float(x["tracking_error"]) for x in universe)
        roe_vals = sorted(float(x["roe"]) for x in universe)

        def q(vals: list[float], ratio: float) -> float:
            if not vals:
                return 0.0
            idx = min(max(int(len(vals) * ratio), 0), len(vals) - 1)
            return vals[idx]

        diagnostics["suggested_criteria"] = {
            "pe_percentile_max": round(
                max(criteria["pe_percentile_max"], q(pe_vals, 0.8)), 3
            ),
            "pb_percentile_max": round(
                max(criteria["pb_percentile_max"], q(pb_vals, 0.8)), 3
            ),
            "tracking_error_max": round(
                max(criteria["tracking_error_max"], q(te_vals, 0.8)), 3
            ),
            "roe_min": round(min(criteria["roe_min"], q(roe_vals, 0.2)), 3),
        }
    return results, diagnostics


def _real_strategy_signals(code: str) -> list[dict]:
    params = deepcopy(_strategy_params_cache.get("semi_monthly_rebalance", {}))
    strategy = SemiMonthlyStrategy(params)
    end_date = date.today()
    start_date = end_date - timedelta(days=120)
    df, _ = _ensure_real_etf_data(code, start_date, end_date)
    if df.empty or "收盘" not in df.columns:
        return []
    prices = pd.to_numeric(df["收盘"], errors="coerce")
    dates = _to_date_series(df["日期"])
    returns = prices.pct_change().fillna(0.0)
    portfolio = Portfolio(initial_cash=100000)
    first_price = float(prices.dropna().iloc[0]) if not prices.dropna().empty else 1.0
    portfolio.add_lot(
        code, shares=1000, cost_per_share=first_price, buy_date=dates.iloc[0]
    )

    out: list[dict] = []
    running_max = first_price
    for i in range(1, len(df)):
        p = float(prices.iloc[i]) if pd.notna(prices.iloc[i]) else None
        d = dates.iloc[i]
        if p is None or pd.isna(d):
            continue
        running_max = max(running_max, p)
        dd = (p - running_max) / running_max if running_max > 0 else 0.0
        hist = prices.iloc[max(0, i - 60) : i + 1].dropna()
        val_pct = float((hist < p).mean()) if not hist.empty else 0.5
        mkt = {
            "prices": {code: p},
            "daily_returns": {code: float(returns.iloc[i])},
            "valuation_percentiles": {code: val_pct},
            "portfolio_drawdown": float(dd),
            "trade_count_today": 0,
        }
        signals = strategy.generate_signals(mkt, portfolio, d)
        for s in signals:
            out.append(
                {
                    "date": d.isoformat(),
                    "signal": s.signal_type.value,
                    "reason": s.reason,
                }
            )
            if len(out) >= 12:
                return out
    return out


def _build_price_data_from_store(
    start_date: date, end_date: date, codes: list[str]
) -> tuple[pd.DataFrame, str]:
    merged: pd.DataFrame | None = None
    sources = set()

    for code in codes:
        df, source = _ensure_real_etf_data(code, start_date, end_date)
        if df.empty or "收盘" not in df.columns:
            continue
        sources.add(source)
        part = df[["日期", "收盘"]].copy()
        part[code] = pd.to_numeric(part["收盘"], errors="coerce")
        part = part[["日期", code]]
        merged = part if merged is None else merged.merge(part, on="日期", how="inner")

    if merged is None or merged.empty:
        return pd.DataFrame(), "synthetic"

    merged = merged.sort_values("日期").reset_index(drop=True)
    if len(merged) < 20:
        return pd.DataFrame(), "synthetic"

    source = "live_akshare" if "live_akshare" in sources else "local_csv"
    return merged, source


def _generate_price_data(
    start_date: date,
    end_date: date,
    codes: list[str],
    seed: int = 20260212,
) -> pd.DataFrame:
    rnd = Random(seed)
    rows = []
    current = start_date
    base = {code: 1.0 + i * 0.02 for i, code in enumerate(codes)}
    while current <= end_date:
        if current.weekday() < 5:
            row: dict[str, Any] = {"日期": current}
            for code in codes:
                drift = 0.0002 + (hash(code) % 7) * 0.00003
                shock = (rnd.random() - 0.5) * 0.01
                base[code] = max(base[code] * (1 + drift + shock), 0.2)
                row[code] = round(base[code], 4)
            rows.append(row)
        current += timedelta(days=1)
    return pd.DataFrame(rows)


def _build_strategy_params(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    params = deepcopy(_strategy_params_cache.get("semi_monthly_rebalance", {}))
    params.setdefault("target_weights", {"510300": 0.4, "510500": 0.3, "518880": 0.3})
    params.setdefault("min_cash_ratio", 0.05)
    params.setdefault("max_trades_per_day", 4)
    if overrides:
        for key, value in overrides.items():
            params[key] = value
    return params


def _run_real_backtest(
    initial_capital: float,
    start_date: date,
    end_date: date,
    benchmark: str,
    strategy_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    codes = ["510300", "510500", "518880"]
    data, data_source = _build_price_data_from_store(start_date, end_date, codes)
    if data.empty:
        data = _generate_price_data(start_date, end_date, codes)
        data_source = "synthetic"
    cfg = BacktestConfig(
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        fee_schedule=FeeSchedule(),
        benchmark_code=benchmark,
    )
    strategy = SemiMonthlyStrategy(_build_strategy_params(strategy_overrides))
    engine = BacktestEngine(config=cfg)
    result = engine.run(strategy=strategy, price_data=data)
    metrics = calculate_backtest_metrics(
        equity_curve=result["equity_curve"],
        trade_log=result["trade_log"],
        initial_capital=initial_capital,
    )
    return {
        "metrics": metrics,
        "data_source": data_source,
        "equity_curve": [
            {
                "date": e["date"].isoformat()
                if hasattr(e["date"], "isoformat")
                else e["date"],
                "total_value": round(float(e["total_value"]), 2),
                "cash": round(float(e["cash"]), 2),
            }
            for e in result["equity_curve"]
        ],
        "trade_log": [
            {
                **t,
                "date": t["date"].isoformat()
                if hasattr(t.get("date"), "isoformat")
                else t.get("date"),
            }
            for t in result["trade_log"]
        ],
    }


def create_app(config: dict | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(_PROJECT_ROOT / "templates" / "web"),
        static_folder=str(_PROJECT_ROOT / "static"),
    )
    app.config["TESTING"] = False
    if config:
        app.config.update(config)

    global _allow_synthetic_fallback, _enable_spot_name_lookup
    _allow_synthetic_fallback = bool(
        app.config.get("TESTING", False)
        or app.config.get("ALLOW_SYNTHETIC_FALLBACK", False)
    )
    _enable_spot_name_lookup = bool(app.config.get("ENABLE_SPOT_NAME_LOOKUP", False))

    _load_strategy_params()
    _register_routes(app)
    return app


def _load_strategy_params() -> None:
    global _strategy_params_cache
    params_path = _CONFIG_DIR / "strategy_params.yaml"
    if params_path.exists():
        _strategy_params_cache = load_config(str(params_path))
    else:
        _strategy_params_cache = {}


def _parse_dt(value: str, default: date) -> date:
    if not value:
        return default
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return default


def _mock_equity_curve(
    start_date: date, end_date: date, initial_capital: float
) -> list[dict]:
    points = []
    current = start_date
    rnd = Random(20260211)
    value = initial_capital
    while current <= end_date:
        drift = 0.00035
        shock = (rnd.random() - 0.5) * 0.01
        value = value * (1 + drift + shock)
        if current.weekday() < 5:
            points.append({"date": current.isoformat(), "total_value": round(value, 2)})
        current = current + timedelta(days=1)
    return points


def _register_routes(app: Flask) -> None:
    @app.route("/api/health")
    def health():
        return jsonify(ApiResponse.success(data={"status": "ok"}).to_dict())

    @app.route("/api/system/status")
    def system_status():
        try:
            ds_cfg_path = _CONFIG_DIR / "data_sources.yaml"
            ds_cfg = load_config(str(ds_cfg_path)) if ds_cfg_path.exists() else {}
            data_source = {
                "primary": (ds_cfg.get("primary") or {}).get("name", "未配置"),
                "fallback_count": len(ds_cfg.get("fallback", []) or []),
                "status": "正常",
            }
            updater_cfg = ds_cfg.get("updater", {}) if isinstance(ds_cfg, dict) else {}
            now = datetime.now()
            full_day = str(updater_cfg.get("full_check_day", "Sunday"))
            weekly_full_check = {
                "enabled": bool(updater_cfg.get("weekly_full_check", False)),
                "full_check_day": full_day,
                "last_check": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "next_check": (now + timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S"),
            }
            cache_status = {
                "status": "可用",
                "last_refresh": _last_refresh_at or now.strftime("%Y-%m-%d %H:%M:%S"),
                "hint": "点击“刷新数据”可拉取最新状态",
            }
            return jsonify(
                ApiResponse.success(
                    data={
                        "data_source": data_source,
                        "weekly_full_check": weekly_full_check,
                        "cache": cache_status,
                    }
                ).to_dict()
            )
        except Exception as e:
            return jsonify(ApiResponse.error(str(e)).to_dict()), 500

    @app.route("/api/system/refresh", methods=["POST"])
    def system_refresh():
        global _last_refresh_at
        try:
            body = request.get_json(silent=True) or {}
            scope = str(body.get("scope", "all"))
            _last_refresh_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return jsonify(
                ApiResponse.success(
                    data={
                        "refresh_requested": True,
                        "scope": scope,
                        "refreshed_at": _last_refresh_at,
                    },
                    message="刷新任务已提交",
                ).to_dict()
            )
        except Exception as e:
            return jsonify(ApiResponse.error(str(e)).to_dict()), 500

    @app.route("/api/backtest/run", methods=["POST"])
    def run_backtest():
        try:
            body = request.get_json(silent=True) or {}
            initial_capital = float(body.get("initial_capital", 100000))
            if initial_capital <= 0:
                return jsonify(ApiResponse.error("初始资金必须大于0").to_dict()), 400

            start_date = _parse_dt(
                body.get("start_date", "2020-01-01"), date(2020, 1, 1)
            )
            end_date = _parse_dt(body.get("end_date", "2025-01-01"), date(2025, 1, 1))
            benchmark = body.get("benchmark", "000300")
            if start_date > end_date:
                return jsonify(
                    ApiResponse.error("开始日期不能晚于结束日期").to_dict()
                ), 400

            result = _run_real_backtest(
                initial_capital=initial_capital,
                start_date=start_date,
                end_date=end_date,
                benchmark=benchmark,
            )
            result["config"] = {
                "initial_capital": initial_capital,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "benchmark": benchmark,
                "data_source": result.get("data_source", "synthetic"),
            }
            if (
                result["config"]["data_source"] == "synthetic"
                and not _allow_synthetic_fallback
            ):
                return jsonify(
                    ApiResponse.error(
                        "当前无真实行情数据，请先刷新数据或检查网络"
                    ).to_dict()
                ), 400
            return jsonify(ApiResponse.success(data=result).to_dict())
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return jsonify(ApiResponse.error(str(e)).to_dict()), 500

    @app.route("/api/backtest/validate", methods=["POST"])
    def validate_backtest():
        try:
            body = request.get_json(silent=True) or {}
            train_ratio = float(body.get("train_ratio", 0.7))
            if train_ratio <= 0 or train_ratio >= 1:
                return jsonify(
                    ApiResponse.error("训练集比例需在0~1之间").to_dict()
                ), 400
            initial_capital = float(body.get("initial_capital", 100000))
            start_date = _parse_dt(
                body.get("start_date", "2020-01-01"), date(2020, 1, 1)
            )
            end_date = _parse_dt(body.get("end_date", "2025-01-01"), date(2025, 1, 1))
            benchmark = str(body.get("benchmark", "000300"))
            all_days = max((end_date - start_date).days, 2)
            split_days = max(1, int(all_days * train_ratio))
            split_date = start_date + timedelta(days=split_days)
            split_date = min(split_date, end_date - timedelta(days=1))
            train_result = _run_real_backtest(
                initial_capital=initial_capital,
                start_date=start_date,
                end_date=split_date,
                benchmark=benchmark,
            )
            validation_result = _run_real_backtest(
                initial_capital=initial_capital,
                start_date=split_date + timedelta(days=1),
                end_date=end_date,
                benchmark=benchmark,
            )
            train_ann = float(train_result["metrics"].get("annual_return", 0.0))
            val_ann = float(validation_result["metrics"].get("annual_return", 0.0))
            denom = max(abs(train_ann), 1e-9)
            stability = max(0.0, 1.0 - abs(train_ann - val_ann) / denom)
            result = {
                "train_ratio": train_ratio,
                "validation_ratio": round(1 - train_ratio, 2),
                "train_annual_return": train_ann,
                "validation_annual_return": val_ann,
                "stability_score": round(stability, 4),
                "train_metrics": train_result["metrics"],
                "validation_metrics": validation_result["metrics"],
                "data_source": train_result.get("data_source", "synthetic"),
            }
            if result["data_source"] == "synthetic" and not _allow_synthetic_fallback:
                return jsonify(
                    ApiResponse.error("样本外验证需要真实行情数据").to_dict()
                ), 400
            return jsonify(ApiResponse.success(data=result).to_dict())
        except Exception as e:
            return jsonify(ApiResponse.error(str(e)).to_dict()), 500

    @app.route("/api/backtest/sensitivity", methods=["POST"])
    def sensitivity_analysis():
        try:
            body = request.get_json(silent=True) or {}
            param_name = str(body.get("param", "deviation_single"))
            values = body.get("values", [0.02, 0.03, 0.04, 0.05, 0.06])
            if not isinstance(values, list) or len(values) == 0:
                return jsonify(ApiResponse.error("参数值列表不能为空").to_dict()), 400
            initial_capital = float(body.get("initial_capital", 100000))
            start_date = _parse_dt(
                body.get("start_date", "2020-01-01"), date(2020, 1, 1)
            )
            end_date = _parse_dt(body.get("end_date", "2025-01-01"), date(2025, 1, 1))
            benchmark = str(body.get("benchmark", "000300"))
            points = []
            for value in values:
                result = _run_real_backtest(
                    initial_capital=initial_capital,
                    start_date=start_date,
                    end_date=end_date,
                    benchmark=benchmark,
                    strategy_overrides={param_name: float(value)},
                )
                points.append(
                    {
                        "value": float(value),
                        "annual_return": float(
                            result["metrics"].get("annual_return", 0.0)
                        ),
                        "data_source": result.get("data_source", "synthetic"),
                    }
                )
            if (
                any(p.get("data_source") == "synthetic" for p in points)
                and not _allow_synthetic_fallback
            ):
                return jsonify(
                    ApiResponse.error("敏感性分析需要真实行情数据").to_dict()
                ), 400
            return jsonify(
                ApiResponse.success(
                    data={"param": param_name, "points": points}
                ).to_dict()
            )
        except Exception as e:
            return jsonify(ApiResponse.error(str(e)).to_dict()), 500

    @app.route("/api/dca/run", methods=["POST"])
    def run_dca():
        try:
            body = request.get_json(silent=True) or {}
            code = body.get("code", "510300")
            amount = float(body.get("amount", 1000))
            start_date = _parse_dt(
                body.get("start_date", "2024-01-01"), date(2024, 1, 1)
            )
            end_date = _parse_dt(body.get("end_date", "2024-12-31"), date(2024, 12, 31))
            interval_days = int(body.get("interval_days", 30))
            mode = body.get("mode", "fixed_amount")

            if amount <= 0:
                return jsonify(ApiResponse.error("定投金额必须大于0").to_dict()), 400
            if interval_days <= 0:
                return jsonify(ApiResponse.error("定投周期必须大于0天").to_dict()), 400

            schedule = FeeSchedule()
            simulator = DCASimulator(
                fee_calculator=FeeCalculator(schedule),
                dca_amount=amount,
                dca_mode=mode,
                dca_dates_or_interval=interval_days,
            )

            nav_data = {}
            data_df, source = _ensure_real_etf_data(code, start_date, end_date)
            if not data_df.empty and "收盘" in data_df.columns:
                for _, row in data_df.iterrows():
                    dt = row["日期"]
                    if isinstance(dt, pd.Timestamp):
                        dt = dt.date()
                    nav_data[dt] = float(row["收盘"])
            else:
                source = "synthetic"
                current = start_date
                index = 0
                while current <= end_date:
                    if current.weekday() < 5:
                        nav_data[current] = round(
                            1.0 + index * 0.002 + ((index % 9) - 4) * 0.003, 4
                        )
                        index += 1
                    current = current + timedelta(days=1)

            if not nav_data:
                return jsonify(
                    ApiResponse.error("日期范围内无可用交易日").to_dict()
                ), 400

            result = simulator.simulate(
                price_series=nav_data, start_date=start_date, end_date=end_date
            )
            dividend_mode = str(body.get("dividend_mode", "cash"))
            return jsonify(
                ApiResponse.success(
                    data={
                        "code": code,
                        "total_invested": result.total_invested,
                        "total_shares": result.total_shares,
                        "avg_cost": result.avg_cost,
                        "current_value": result.current_value,
                        "total_return": result.total_return,
                        "dividend_mode": dividend_mode,
                        "data_source": source,
                        "purchase_records": result.purchase_records,
                    }
                ).to_dict()
            )
        except Exception as e:
            logger.error(f"DCA error: {e}")
            return jsonify(ApiResponse.error(str(e)).to_dict()), 500

    @app.route("/api/data/etf/<code>")
    def get_etf_data(code: str):
        try:
            start_date = _parse_dt(
                request.args.get("start", "2024-01-01"), date(2024, 1, 1)
            )
            end_date = _parse_dt(
                request.args.get("end", "2024-12-31"), date(2024, 12, 31)
            )
            records = []
            data_df, source = _ensure_real_etf_data(code, start_date, end_date)
            if not data_df.empty:
                pct_col = data_df.get("涨跌幅")
                for _, row in data_df.iterrows():
                    pct_raw = 0.0
                    if pct_col is not None:
                        pct_raw = float(
                            pd.to_numeric(row.get("涨跌幅", 0.0), errors="coerce")
                            or 0.0
                        )
                    pct = pct_raw / 100.0 if abs(pct_raw) > 1 else pct_raw
                    dt = row.get("日期")
                    if isinstance(dt, pd.Timestamp):
                        dt = dt.date()
                    if dt is None:
                        continue
                    nav_val = float(
                        pd.to_numeric(row.get("收盘", 0.0), errors="coerce") or 0.0
                    )
                    vol_val = int(
                        float(
                            pd.to_numeric(row.get("成交量", 0.0), errors="coerce")
                            or 0.0
                        )
                    )
                    records.append(
                        {
                            "date": dt.isoformat()
                            if hasattr(dt, "isoformat")
                            else str(dt),
                            "nav": nav_val,
                            "pct_change": float(pct),
                            "volume": vol_val,
                        }
                    )
            else:
                source = "synthetic"
                if not _allow_synthetic_fallback:
                    return jsonify(
                        ApiResponse.error(
                            "未获取到真实行情数据，请稍后重试或先刷新数据"
                        ).to_dict()
                    ), 400
                current = start_date
                idx = 0
                while current <= end_date:
                    if current.weekday() < 5:
                        nav = round(1.0 + idx * 0.0015 + ((idx % 7) - 3) * 0.002, 4)
                        records.append(
                            {
                                "date": current.isoformat(),
                                "nav": nav,
                                "pct_change": round(((idx % 7) - 3) * 0.004, 4),
                                "volume": 1000000 + idx * 2500,
                            }
                        )
                        idx += 1
                    current = current + timedelta(days=1)
            result = {
                "code": code,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "source": source,
                "cache_hit": source != "live_akshare",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "records": records,
            }
            return jsonify(ApiResponse.success(data=result).to_dict())
        except Exception as e:
            logger.error(f"ETF data error: {e}")
            return jsonify(ApiResponse.error(str(e)).to_dict()), 500

    @app.route("/api/selection/screen", methods=["POST"])
    def screen_funds():
        try:
            body = request.get_json(silent=True) or {}
            criteria = {
                "min_size": float(body.get("min_size", 1e8)),
                "max_fee": float(body.get("max_fee", 0.005)),
                "categories": body.get("categories", []),
                "pe_percentile_max": float(body.get("pe_percentile_max", 0.35)),
                "pb_percentile_max": float(body.get("pb_percentile_max", 0.45)),
                "roe_min": float(body.get("roe_min", 0.08)),
                "tracking_error_max": float(body.get("tracking_error_max", 0.03)),
            }
            results, diagnostics = _screen_real_funds(criteria)
            if not results:
                return jsonify(
                    ApiResponse.success(
                        data={
                            "criteria": criteria,
                            "results": [],
                            "total": 0,
                            "hint": "当前筛选条件较严格，建议使用下方推荐参数一键重试",
                            "diagnostics": diagnostics,
                        },
                        message="暂无满足条件的真实基金数据",
                    ).to_dict()
                )

            return jsonify(
                ApiResponse.success(
                    data={
                        "criteria": criteria,
                        "results": results,
                        "total": len(results),
                        "diagnostics": diagnostics,
                    }
                ).to_dict()
            )
        except Exception as e:
            logger.error(f"Screening error: {e}")
            return jsonify(ApiResponse.error(str(e)).to_dict()), 500

    @app.route("/api/strategy/params", methods=["GET", "PUT"])
    def strategy_params():
        global _strategy_params_cache
        try:
            if request.method == "GET":
                return jsonify(
                    ApiResponse.success(data=_strategy_params_cache).to_dict()
                )
            body = request.get_json(silent=True) or {}
            if not body:
                return jsonify(ApiResponse.error("请求体为空").to_dict()), 400
            _strategy_params_cache.update(body)
            return jsonify(
                ApiResponse.success(
                    data=_strategy_params_cache, message="更新成功"
                ).to_dict()
            )
        except Exception as e:
            logger.error(f"Strategy params error: {e}")
            return jsonify(ApiResponse.error(str(e)).to_dict()), 500

    @app.route("/api/strategy/signals", methods=["POST"])
    def strategy_signals():
        try:
            body = request.get_json(silent=True) or {}
            code = body.get("code", "510300")
            signals = _real_strategy_signals(code)
            if not signals and not _allow_synthetic_fallback:
                return jsonify(
                    ApiResponse.error("暂无可生成的真实策略信号").to_dict()
                ), 400
            return jsonify(
                ApiResponse.success(data={"code": code, "signals": signals}).to_dict()
            )
        except Exception as e:
            return jsonify(ApiResponse.error(str(e)).to_dict()), 500

    @app.route("/")
    def dashboard():
        try:
            return render_template("dashboard.html")
        except TemplateNotFound:
            return jsonify(ApiResponse.error("仪表盘页面不存在").to_dict()), 404

    @app.route("/backtest")
    def backtest_page():
        try:
            return render_template("backtest.html")
        except TemplateNotFound:
            return jsonify(ApiResponse.error("回测页面不存在").to_dict()), 404

    @app.route("/screening")
    def screening_page():
        try:
            return render_template("screening.html")
        except TemplateNotFound:
            return jsonify(ApiResponse.error("筛选页面不存在").to_dict()), 404

    @app.route("/strategy")
    def strategy_page():
        try:
            return render_template("strategy.html")
        except TemplateNotFound:
            return jsonify(ApiResponse.error("策略页面不存在").to_dict()), 404

    @app.route("/data")
    def data_page():
        try:
            return render_template("data.html")
        except TemplateNotFound:
            return jsonify(ApiResponse.error("数据页面不存在").to_dict()), 404

    @app.route("/dca")
    def dca_page():
        try:
            return render_template("dca.html")
        except TemplateNotFound:
            return jsonify(ApiResponse.error("定投页面不存在").to_dict()), 404
