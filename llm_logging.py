from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional
import threading

LLM_IO_LOCK = threading.Lock()


def log_llm_call(
    llm_io_path: Path,
    *,
    request_id: str,
    telegram_user_id: str,
    telegram_username: str,
    pdf_filename: str,
    model: str,
    api_key_id: str,
    request_text: str,
    response_text: str,
    latency_seconds: Optional[float],
    status: str,
    error_type: Optional[str] = None,
    error_code: Optional[str] = None,
) -> None:
    """
    Записывает один вызов ЛЛМ в llm_io_...txt в человекочитаемом виде.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    request_text = request_text or ""
    response_text = response_text or ""

    request_chars = len(request_text)
    response_chars = len(response_text)
    latency_str = "N/A" if latency_seconds is None else f"{latency_seconds:.3f}"

    llm_io_path.parent.mkdir(parents=True, exist_ok=True)

    with LLM_IO_LOCK:
        with llm_io_path.open("a", encoding="utf-8") as f:
            f.write("===== LLM CALL START =====\n")
            f.write("[META]\n")
            f.write(f"date: {date_str}\n")
            f.write(f"time: {time_str}\n")
            f.write(f"request_id: {request_id}\n")
            f.write(f"telegram_user_id: {telegram_user_id}\n")
            f.write(f"telegram_username: {telegram_username}\n")
            f.write(f"pdf_filename: {pdf_filename}\n")
            f.write(f"model: {model}\n")
            f.write(f"api_key_id: {api_key_id}\n")
            f.write(f"request_chars: {request_chars}\n")
            f.write(f"response_chars: {response_chars}\n")
            f.write(f"latency_seconds: {latency_str}\n")
            f.write(f"status: {status}\n")
            if error_type:
                f.write(f"error_type: {error_type}\n")
            if error_code:
                f.write(f"error_code: {error_code}\n")

            f.write("\n[REQUEST]\n")
            if request_text:
                f.write(request_text)
                if not request_text.endswith("\n"):
                    f.write("\n")
            else:
                f.write("N/A\n")

            f.write("\n[RESPONSE]\n")
            if response_text:
                f.write(response_text)
                if not response_text.endswith("\n"):
                    f.write("\n")
            else:
                f.write("N/A\n")

            f.write("===== LLM CALL END =====\n\n")

