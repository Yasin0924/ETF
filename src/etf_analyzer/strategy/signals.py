"""Trading signal types."""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    ADD = "add"
    HOLD = "hold"
    REBALANCE = "rebalance"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"


@dataclass
class Signal:
    signal_type: SignalType
    code: str
    reason: str
    strength: float = 1.0
    target_amount: float = 0.0
    target_weight: float = 0.0
    date: Optional[date] = None
