from __future__ import annotations

import tarfile
from datetime import datetime
from pathlib import Path
from logging import Logger

from telegram import Update
from telegram.ext import ContextTypes, Application, CommandHandler

from config import DATA_DIR, LOG_DIR


MAX_ARCHIVE_SIZE_BYTES = 50 * 1024 * 1024  # 50 МБ

USER_ARCHIVES_TOO_LARGE: set[str] = set()
USER_DIRS_WITH_DATA: set[str] = set()


class ArchiveTooLargeError(Exception):
    """
    Исключение, сигнализирующее, что архив превысил максимально допустимый размер.
    """
    pass


def _get_user_context(update: Update) -> dict[str, object]:
    """
    Локальный хелпер контекста пользователя для логирования.
    """
    user = update.effective_user
    if user is None:
        return {
            "telegram_user_id": "N/A",
            "telegram_username": "N/A",
        }
    return {
        "telegram_user_id": user.id,
        "telegram_username": user.username or "N/A",
    }


def create_logs_archive() -> Path | None:
    """
    Создаёт архив со всеми файлами и подкаталогами в LOG_DIR (data/logfiles).

    Возвращает:
      - Path к архиву, если он успешно создан и не превышает лимит;
      - None, если LOG_DIR не существует или пуст.

    Бросает ArchiveTooLargeError, если размер архива превышает лимит.
    """
    if not LOG_DIR.exists():
        return None

    has_entries = any(LOG_DIR.iterdir())
    if not has_entries:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = LOG_DIR.parent / f"logs_{timestamp}.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(LOG_DIR, arcname=LOG_DIR.name)

    size = archive_path.stat().st_size
    if size > MAX_ARCHIVE_SIZE_BYTES:
        archive_path.unlink(missing_ok=True)
        raise ArchiveTooLargeError(
            f"Logs archive {archive_path} is larger than allowed limit."
        )

    return archive_path


def create_user_archives() -> list[Path]:
    """
    Создаёт архивы с последними до 5 запросов по каждому пользователю.

    Логика:
      - Ищет в DATA_DIR подкаталоги, имена которых состоят только из цифр — это user_id.
      - Для каждого user_id:
          * собирает его подпапки одного уровня глубины;
          * сортирует по имени по убыванию;
          * берёт первые пять (или меньше, если их меньше);
          * упаковывает в архив <user_id>.tar.gz в каталоге DATA_DIR / "archives".
      - Если размер архива > MAX_ARCHIVE_SIZE_BYTES, архив удаляется,
        user_id попадает в USER_ARCHIVES_TOO_LARGE, в результат не добавляется.

    Возвращает список путей ко всем успешно созданным архивам (<= лимита).

    Глобальные множества:
      - USER_DIRS_WITH_DATA — user_id, у которых вообще были подпапки;
      - USER_ARCHIVES_TOO_LARGE — user_id, чьи архивы превысили лимит.
    """
    global USER_ARCHIVES_TOO_LARGE, USER_DIRS_WITH_DATA
    USER_ARCHIVES_TOO_LARGE = set()
    USER_DIRS_WITH_DATA = set()

    archives: list[Path] = []

    if not DATA_DIR.exists():
        return archives

    archives_dir = DATA_DIR / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    for entry in DATA_DIR.iterdir():
        if not entry.is_dir():
            continue

        user_id = entry.name
        if not user_id.isdigit():
            continue

        subdirs = [p for p in entry.iterdir() if p.is_dir()]
        if not subdirs:
            continue

        USER_DIRS_WITH_DATA.add(user_id)

        subdirs_sorted = sorted(subdirs, key=lambda p: p.name, reverse=True)
        selected = subdirs_sorted[:5]

        archive_path = archives_dir / f"{user_id}.tar.gz"

        with tarfile.open(archive_path, "w:gz") as tf:
            for subdir in selected:
                tf.add(subdir, arcname=subdir.name)

        size = archive_path.stat().st_size
        if size > MAX_ARCHIVE_SIZE_BYTES:
            USER_ARCHIVES_TOO_LARGE.add(user_id)
            archive_path.unlink(missing_ok=True)
            continue

        archives.append(archive_path)

    return archives


async def download_logs_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик команды /download_logs.

    Готовит и отправляет:
      - архив со всеми логами из data/logfiles (если он существует и не превышает 50 МБ);
      - архивы по каждому пользователю с последними до 5 запросов (по data/<user_id>/...).

    При превышении лимита 50 МБ по любому архиву отправляет текстовое сообщение
    с объяснением и не отправляет сам архив.
    """
    logger: Logger = context.application.bot_data["logger"]
    user_ctx = _get_user_context(update)
    message = update.message

    if message is None:
        logger.warning(
            "Сообщение в download_logs_handler без message.",
            extra={
                "stage": "download_logs_no_message",
                **user_ctx,
                "request_id": "N/A",
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": "N/A",
            },
        )
        return

    try:
        logger.info(
            "Получена команда /download_logs.",
            extra={
                "stage": "download_logs_command",
                **user_ctx,
                "request_id": "N/A",
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": "N/A",
            },
        )

        await message.reply_text("Готовлю архивы с логами, подождите...")

        # 1) Архив с каталогом data/logfiles
        logs_archive_path: Path | None = None
        try:
            logs_archive_path = create_logs_archive()
        except ArchiveTooLargeError:
            logger.warning(
                "Архив с логами из data/logfiles превышает 50 МБ и не будет отправлен.",
                extra={
                    "stage": "download_logs_logs_too_big",
                    **user_ctx,
                    "request_id": "N/A",
                    "duration_seconds": 0,
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": "N/A",
                },
            )
            await message.reply_text(
                "Архив с логами из data/logfiles больше 50 МБ — не могу отправить его в Telegram. "
                "Скачайте логи вручную с сервера."
            )
        except Exception as e:
            logger.error(
                "Ошибка при создании архива с логами data/logfiles: %s",
                e,
                exc_info=True,
                extra={
                    "stage": "download_logs_logs_error",
                    **user_ctx,
                    "request_id": "N/A",
                    "duration_seconds": 0,
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": "N/A",
                },
            )
            logs_archive_path = None

        if logs_archive_path is None:
            logger.info(
                "Каталог data/logfiles отсутствует или пуст, архив логов не создаётся.",
                extra={
                    "stage": "download_logs_logs_missing",
                    **user_ctx,
                    "request_id": "N/A",
                    "duration_seconds": 0,
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": "N/A",
                },
            )
        else:
            try:
                with logs_archive_path.open("rb") as f:
                    await message.reply_document(
                        document=f,
                        filename=logs_archive_path.name,
                        caption="Все логи бота (data/logfiles)",
                    )
                logger.info(
                    "Архив с логами data/logfiles отправлен пользователю.",
                    extra={
                        "stage": "download_logs_logs_archived",
                        **user_ctx,
                        "request_id": "N/A",
                        "duration_seconds": 0,
                        "model": "N/A",
                        "api_key_id": "N/A",
                        "file_name": logs_archive_path.name,
                    },
                )
            finally:
                try:
                    logs_archive_path.unlink(missing_ok=True)
                except Exception:
                    logger.warning(
                        "Не удалось удалить временный архив логов %s.",
                        logs_archive_path,
                        extra={
                            "stage": "download_logs_logs_cleanup_warning",
                            **user_ctx,
                            "request_id": "N/A",
                            "duration_seconds": 0,
                            "model": "N/A",
                            "api_key_id": "N/A",
                            "file_name": logs_archive_path.name,
                        },
                    )

        # 2) Архивы по каждому user_id в data/
        logger.info(
            "Начинаю формирование архивов по пользователям из каталога data.",
            extra={
                "stage": "download_logs_user_archives_start",
                **user_ctx,
                "request_id": "N/A",
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": "N/A",
            },
        )

        user_archives = create_user_archives()

        if not USER_DIRS_WITH_DATA:
            logger.info(
                "В каталоге data нет директорий с запросами пользователей.",
                extra={
                    "stage": "download_logs_user_archives_none",
                    **user_ctx,
                    "request_id": "N/A",
                    "duration_seconds": 0,
                    "model": "N/A",
                    "api_key_id": "N/A",
                    "file_name": "N/A",
                },
            )
            await message.reply_text(
                "В каталоге data пока нет директорий с запросами пользователей."
            )
        else:
            # Сообщения для пользователей, чьи архивы превысили лимит
            if USER_ARCHIVES_TOO_LARGE:
                for user_id in sorted(USER_ARCHIVES_TOO_LARGE):
                    logger.warning(
                        "Архив с последними запросами пользователя %s превышает 50 МБ и не был отправлен.",
                        user_id,
                        extra={
                            "stage": "download_logs_user_archive_too_big",
                            **user_ctx,
                            "request_id": "N/A",
                            "duration_seconds": 0,
                            "model": "N/A",
                            "api_key_id": "N/A",
                            "file_name": "N/A",
                        },
                    )
                    await message.reply_text(
                        f"Архив с последними запросами пользователя {user_id} больше 50 МБ и не был отправлен."
                    )

            # Отправка успешно созданных архивов
            for archive_path in user_archives:
                user_id = archive_path.stem
                try:
                    with archive_path.open("rb") as f:
                        await message.reply_document(
                            document=f,
                            filename=archive_path.name,
                            caption=f"Последние до 5 запросов пользователя {user_id}.",
                        )
                    logger.info(
                        "Архив запросов пользователя %s отправлен.",
                        user_id,
                        extra={
                            "stage": "download_logs_user_archive_sent",
                            **user_ctx,
                            "request_id": "N/A",
                            "duration_seconds": 0,
                            "model": "N/A",
                            "api_key_id": "N/A",
                            "file_name": archive_path.name,
                        },
                    )
                finally:
                    try:
                        archive_path.unlink(missing_ok=True)
                    except Exception:
                        logger.warning(
                            "Не удалось удалить временный архив запросов %s.",
                            archive_path,
                            extra={
                                "stage": "download_logs_user_archive_cleanup_warning",
                                **user_ctx,
                                "request_id": "N/A",
                                "duration_seconds": 0,
                                "model": "N/A",
                                "api_key_id": "N/A",
                                "file_name": archive_path.name,
                            },
                        )

    except Exception as e:
        logger.error(
            "Ошибка при выполнении команды /download_logs: %s",
            e,
            exc_info=True,
            extra={
                "stage": "download_logs_error",
                **user_ctx,
                "request_id": "N/A",
                "duration_seconds": 0,
                "model": "N/A",
                "api_key_id": "N/A",
                "file_name": "N/A",
            },
        )
        await message.reply_text(
            "Во время подготовки архивов логов произошла ошибка. "
            "Попробуйте позже или свяжитесь с разработчиком @Sergey_robots."
        )


