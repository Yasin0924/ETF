"""Unified API response wrapper (status_code + data + message)."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class StatusCode(IntEnum):
    SUCCESS = 0
    WARNING = 1
    ERROR = 2


@dataclass(frozen=True)
class ApiResponse:
    """All module interfaces return this to enable unified error handling."""

    status_code: StatusCode
    data: Any = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status_code != StatusCode.ERROR

    @classmethod
    def success(cls, data: Any = None, message: str = "") -> "ApiResponse":
        return cls(status_code=StatusCode.SUCCESS, data=data, message=message)

    @classmethod
    def error(cls, message: str, data: Any = None) -> "ApiResponse":
        return cls(status_code=StatusCode.ERROR, data=data, message=message)

    @classmethod
    def warning(cls, data: Any = None, message: str = "") -> "ApiResponse":
        return cls(status_code=StatusCode.WARNING, data=data, message=message)

    def to_dict(self) -> dict:
        return {
            "status_code": self.status_code.value,
            "data": self.data,
            "message": self.message,
        }
