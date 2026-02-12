"""Tests for A-share trading calendar."""

from datetime import date
from etf_analyzer.core.calendar import (
    is_trading_day,
    next_trading_day,
    get_trading_days,
)


class TestTradingCalendar:
    def test_weekday_is_trading_day(self):
        # 2024-01-02 is a Tuesday
        assert is_trading_day(date(2024, 1, 2)) is True

    def test_weekend_is_not_trading_day(self):
        # 2024-01-06 is a Saturday
        assert is_trading_day(date(2024, 1, 6)) is False
        # 2024-01-07 is a Sunday
        assert is_trading_day(date(2024, 1, 7)) is False

    def test_next_trading_day_from_friday(self):
        # 2024-01-05 is a Friday -> next trading day is Monday 2024-01-08
        result = next_trading_day(date(2024, 1, 5))
        assert result == date(2024, 1, 8)

    def test_next_trading_day_from_weekday(self):
        result = next_trading_day(date(2024, 1, 2))
        assert result == date(2024, 1, 3)

    def test_get_trading_days_range(self):
        days = get_trading_days(date(2024, 1, 1), date(2024, 1, 7))
        # Jan 1 Mon(holiday-like but we just check weekdays for now)
        # Jan 2 Tue, Jan 3 Wed, Jan 4 Thu, Jan 5 Fri => 5 weekdays incl Jan 1
        assert all(d.weekday() < 5 for d in days)
        assert len(days) == 5  # Mon-Fri

    def test_t_plus_n_settlement(self):
        """T+1 confirmation: buy on Tuesday, confirmed on Wednesday."""
        from etf_analyzer.core.calendar import settle_date

        buy_date = date(2024, 1, 2)  # Tuesday
        assert settle_date(buy_date, n=1) == date(2024, 1, 3)

    def test_t_plus_n_over_weekend(self):
        """T+2 redemption: sell on Thursday, arrives Monday."""
        from etf_analyzer.core.calendar import settle_date

        sell_date = date(2024, 1, 4)  # Thursday
        assert settle_date(sell_date, n=2) == date(2024, 1, 8)  # Monday
