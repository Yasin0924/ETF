"""Backtest visualization: equity curves, drawdown charts."""

import base64
from io import BytesIO
from typing import Optional
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def plot_equity_curve(
    dates: pd.Series,
    portfolio_values: pd.Series,
    benchmark_values: Optional[pd.Series] = None,
    title: str = "Cumulative Returns",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, portfolio_values, label="Portfolio", linewidth=1.5, color="#2196F3")
    if benchmark_values is not None:
        ax.plot(
            dates,
            benchmark_values,
            label="Benchmark",
            linewidth=1.0,
            color="#9E9E9E",
            linestyle="--",
        )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Value (CNY)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_drawdown(
    dates: pd.Series, portfolio_values: pd.Series, title: str = "Drawdown"
) -> plt.Figure:
    values = pd.Series(portfolio_values.values, index=dates)
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dates, drawdown, 0, color="red", alpha=0.3)
    ax.plot(dates, drawdown, color="red", linewidth=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


_PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4", "#795548"]


def plot_position_allocation(
    dates: pd.Series,
    weights: pd.DataFrame,
    title: str = "Position Allocation",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    cols = list(weights.columns)
    totals = weights.sum(axis=1)
    cash_weights = (1.0 - totals).clip(lower=0)
    labels = cols + ["Cash"]
    data = [weights[c].values for c in cols] + [cash_weights.values]
    colors = _PALETTE[: len(cols)] + ["#E0E0E0"]
    ax.stackplot(dates, *data, labels=labels, colors=colors, alpha=0.85)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_trade_signals(
    dates: pd.Series,
    prices: pd.Series,
    trade_log: list,
    title: str = "Trade Signals",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, prices, color="#2196F3", linewidth=1.0, label="NAV")
    buy_types = {"buy", "add"}
    sell_types = {"sell", "take_profit", "stop_loss"}
    buy_dates = [t["date"] for t in trade_log if t.get("type") in buy_types]
    buy_navs = [t["nav"] for t in trade_log if t.get("type") in buy_types]
    sell_dates = [t["date"] for t in trade_log if t.get("type") in sell_types]
    sell_navs = [t["nav"] for t in trade_log if t.get("type") in sell_types]
    if buy_dates:
        ax.scatter(
            buy_dates,
            buy_navs,
            marker="^",
            color="#4CAF50",
            s=60,
            label="Buy",
            zorder=5,
        )
    if sell_dates:
        ax.scatter(
            sell_dates,
            sell_navs,
            marker="v",
            color="#F44336",
            s=60,
            label="Sell",
            zorder=5,
        )
    max_annotations = 8
    for t in trade_log[:max_annotations]:
        ax.annotate(
            t.get("reason", "")[:12],
            xy=(t["date"], t["nav"]),
            fontsize=6,
            textcoords="offset points",
            xytext=(0, 10 if t.get("type") in buy_types else -12),
            ha="center",
        )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("NAV (CNY)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_to_base64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64
