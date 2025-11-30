from __future__ import annotations

from logging import Logger
from pathlib import Path

from config import ensure_directories
from logging_setup import setup_logging
from bot import create_application


def main() -> None:
    ensure_directories()
    logger, log_path = setup_logging()

    start_msg = "Старт системы обработки отчётов НБКИ. Лог: %s"
    logger.info(
        start_msg,
        log_path,
        extra={
            "stage": "system_start",
            "telegram_user_id": "N/A",
            "telegram_username": "N/A",
            "request_id": "N/A",
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": "N/A",
        },
    )

    application = create_application(logger)
    logger.info(
        "Telegram-бот запущен. Ожидаю сообщения...",
        extra={
            "stage": "bot_run_polling",
            "telegram_user_id": "N/A",
            "telegram_username": "N/A",
            "request_id": "N/A",
            "duration_seconds": 0,
            "model": "N/A",
            "api_key_id": "N/A",
            "file_name": "N/A",
        },
    )
    application.run_polling()


if __name__ == "__main__":
    main()
