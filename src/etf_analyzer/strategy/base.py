"""Abstract base strategy class."""

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Dict, List
from etf_analyzer.strategy.signals import Signal
from etf_analyzer.simulation.portfolio import Portfolio


class BaseStrategy(ABC):
    def __init__(self, name: str, params: dict = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def generate_signals(
        self,
        market_data: Dict[str, Any],
        portfolio: Portfolio,
        current_date: date,
    ) -> List[Signal]: ...
