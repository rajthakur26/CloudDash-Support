"""
Structured logging with JSON output and per-conversation trace IDs.
"""
import logging
import sys
import json
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    _SKIP = frozenset((
        "args", "asctime", "created", "exc_info", "exc_text",
        "filename", "funcName", "id", "levelname", "levelno",
        "lineno", "module", "msecs", "message", "msg", "name",
        "pathname", "process", "processName", "relativeCreated",
        "stack_info", "thread", "threadName",
    ))

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._SKIP:
                log_obj[key] = value
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, default=str)


class ConsoleFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m", "INFO": "\033[32m",
        "WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return f"{color}[{ts}] {record.levelname:<8}{self.RESET} {record.name}: {record.getMessage()}"


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if fmt == "json" else ConsoleFormatter())
    root.handlers = [handler]


class StructuredLogger:
    """Wraps standard Logger; allows kwargs as structured fields."""
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, msg: str, **kwargs) -> None:
        exc_info = kwargs.pop("exc_info", None)
        extra = kwargs if kwargs else None
        self._logger.log(level, msg, extra=extra, exc_info=exc_info, stacklevel=3)

    def debug(self, msg: str, **kwargs) -> None:
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs) -> None:
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs) -> None:
        self._log(logging.CRITICAL, msg, **kwargs)

    def exception(self, msg: str, **kwargs) -> None:
        kwargs["exc_info"] = True
        self._log(logging.ERROR, msg, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(logging.getLogger(name))
