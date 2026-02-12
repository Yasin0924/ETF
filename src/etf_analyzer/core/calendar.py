"""A-share trading calendar utilities.

Note: This is a simplified calendar using weekday-only logic.
For production, integrate with exchange holiday data or use
`exchange_calendars` package for accurate holiday handling.
"""

from datetime import date, timedelta
from typing import List


def is_trading_day(d: date) -> bool:
    """Check if date is a trading day (weekday, not holiday).

    Currently only checks weekdays. TODO: add CN holiday support.
    """
    return d.weekday() < 5  # Mon=0 ... Fri=4


def next_trading_day(d: date) -> date:
    """Return the next trading day after given date."""
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def get_trading_days(start: date, end: date) -> List[date]:
    """Return list of trading days in [start, end] inclusive."""
    days = []
    current = start
    while current <= end:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def settle_date(trade_date: date, n: int = 1) -> date:
    """Calculate T+N settlement date (N trading days after trade_date).

    Args:
        trade_date: The trade execution date.
        n: Number of trading days to settle (1 for purchase, 2 for redemption).
    """
    current = trade_date
    for _ in range(n):
        current = next_trading_day(current)
    return current
