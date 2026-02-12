"""HTML report generation using Jinja2 templates."""

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from etf_analyzer.backtest.visualization import (
    plot_equity_curve,
    plot_drawdown,
    fig_to_base64,
)
from etf_analyzer.core.logger import get_logger

logger = get_logger("backtest.report")
_TEMPLATE_DIR = Path(__file__).parent.parent.parent.parent / "templates"


class _DateEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, date):
            return o.isoformat()
        return super().default(o)


def generate_html_report(
    metrics: Dict[str, Any],
    equity_curve: pd.DataFrame,
    output_path: str,
    benchmark_col: Optional[str] = "benchmark",
    template_dir: str = None,
    trade_log: Optional[List[dict]] = None,
    position_weights: Optional[pd.DataFrame] = None,
) -> str:
    tpl_dir = Path(template_dir) if template_dir else _TEMPLATE_DIR
    if not tpl_dir.exists():
        tpl_dir = Path(output_path).parent
        _create_fallback_template(tpl_dir)
    env = Environment(loader=FileSystemLoader(str(tpl_dir)))
    template = env.get_template("report.html")

    equity_chart = None
    drawdown_chart = None
    echarts_equity = None
    echarts_drawdown = None
    echarts_position = None
    trade_log_json = None

    if "date" in equity_curve.columns and "portfolio" in equity_curve.columns:
        benchmark = equity_curve.get(benchmark_col)
        fig_eq = plot_equity_curve(
            dates=equity_curve["date"],
            portfolio_values=equity_curve["portfolio"],
            benchmark_values=benchmark,
        )
        equity_chart = fig_to_base64(fig_eq)
        fig_dd = plot_drawdown(
            dates=equity_curve["date"], portfolio_values=equity_curve["portfolio"]
        )
        drawdown_chart = fig_to_base64(fig_dd)

        dates_str = [
            d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            for d in equity_curve["date"]
        ]
        portfolio_vals = equity_curve["portfolio"].tolist()
        echarts_equity = {
            "dates": dates_str,
            "portfolio": portfolio_vals,
        }
        if benchmark is not None:
            echarts_equity["benchmark"] = benchmark.tolist()

        values = pd.Series(portfolio_vals)
        cummax = values.cummax()
        dd = ((values - cummax) / cummax).tolist()
        echarts_drawdown = {"dates": dates_str, "drawdown": dd}

    if position_weights is not None and "date" in equity_curve.columns:
        pw_dates = [
            d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            for d in equity_curve["date"]
        ]
        pw_data = {}
        for col in position_weights.columns:
            pw_data[col] = position_weights[col].tolist()
        echarts_position = {"dates": pw_dates, "series": pw_data}

    if trade_log:
        trade_log_json = json.dumps(trade_log, cls=_DateEncoder, ensure_ascii=False)

    html = template.render(
        report_date=date.today().strftime("%Y-%m-%d"),
        metrics=metrics,
        equity_chart=equity_chart,
        drawdown_chart=drawdown_chart,
        echarts_equity=json.dumps(echarts_equity) if echarts_equity else None,
        echarts_drawdown=json.dumps(echarts_drawdown) if echarts_drawdown else None,
        echarts_position=json.dumps(echarts_position) if echarts_position else None,
        trade_log=trade_log or [],
        trade_log_json=trade_log_json,
    )
    Path(output_path).write_text(html, encoding="utf-8")
    logger.info(f"Report generated: {output_path}")
    return output_path


def _create_fallback_template(directory: Path) -> None:
    template_path = directory / "report.html"
    if template_path.exists():
        return
    template_path.write_text(
        """<!DOCTYPE html><html><head><title>Backtest Report</title></head>
<body><h1>Backtest Report</h1><p>Generated: {{ report_date }}</p>
<h2>Metrics</h2><table>{% for k, v in metrics.items() %}
<tr><td>{{ k }}</td><td>{{ v }}</td></tr>{% endfor %}</table>
{% if equity_chart %}<h2>Equity</h2><img src="data:image/png;base64,{{ equity_chart }}">{% endif %}
{% if drawdown_chart %}<h2>Drawdown</h2><img src="data:image/png;base64,{{ drawdown_chart }}">{% endif %}
</body></html>""",
        encoding="utf-8",
    )
