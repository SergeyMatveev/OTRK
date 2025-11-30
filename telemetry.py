from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import traceback

from config import LOG_DIR


ERRORS_PATH = LOG_DIR / "errors.txt"
STATS_PATH = LOG_DIR / "stats.csv"


def log_error_to_file(
    logger,
    stage: str,
    exc: Exception,
    *,
    message: str = "",
    request_id: str = "",
    telegram_user_id: Any = "",
    telegram_username: str = "",
) -> None:
    """
    Логирует ошибку в errors.txt и отправляет краткую запись в основной логгер.
    """
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")

    header_line = (
        f"[{ts}] level=ERROR stage={stage} "
        f"request_id={request_id} user_id={telegram_user_id} username={telegram_username}"
    )
    message_line = f"message: {message or str(exc)}"
    exception_line = f"exception: {repr(exc)}"
    traceback_text = traceback.format_exc()
    traceback_line = f"traceback:\n{traceback_text}"

    block = "\n".join([header_line, message_line, exception_line, traceback_line, "----", ""])

    ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ERRORS_PATH.open("a", encoding="utf-8") as f:
        f.write(block)

    # Краткая запись в основной логгер без полного трейсбэка
    logger.error(
        message or str(exc),
        extra={
            "stage": stage,
            "request_id": request_id,
            "telegram_user_id": telegram_user_id,
            "telegram_username": telegram_username,
        },
    )


def append_stats_row(
    date: str,
    time: str,
    telegram_user_id: Any,
    telegram_username: str,
    request_id: str,
    pdf_filename: str,
    pages_original: int,
    pages_after_crop: int,
    processed_successfully: bool,
    model: str,
    api_key_id: str,
    llm_calls_count: int,
    total_request_chars: int,
    total_response_chars: int,
    max_latency_seconds: float,
) -> None:
    """
    Добавляет строку статистики в stats.csv.
    """
    status = "success" if processed_successfully else "error"

    def _to_str(value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    row = [
        date,
        time,
        _to_str(telegram_user_id),
        telegram_username,
        request_id,
        pdf_filename,
        _to_str(pages_original),
        _to_str(pages_after_crop),
        status,
        model,
        api_key_id,
        _to_str(llm_calls_count),
        _to_str(total_request_chars),
        _to_str(total_response_chars),
        _to_str(max_latency_seconds),
    ]

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(row)


def log_info(logger, stage: str, message: str, **extra: Any) -> None:
    """
    Обёртка для logger.info с указанием этапа (stage) и дополнительных полей.
    """
    logger.info(message, extra={"stage": stage, **extra})


def log_debug(logger, stage: str, message: str, **extra: Any) -> None:
    """
    Обёртка для logger.debug с указанием этапа (stage) и дополнительных полей.
    """
    logger.debug(message, extra={"stage": stage, **extra})


def log_warning(logger, stage: str, message: str, **extra: Any) -> None:
    """
    Обёртка для logger.warning с указанием этапа (stage) и дополнительных полей.
    """
    logger.warning(message, extra={"stage": stage, **extra})
