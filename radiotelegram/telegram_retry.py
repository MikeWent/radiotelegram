import time
from dataclasses import dataclass

from requests import exceptions as requests_exceptions
from telebot.apihelper import ApiTelegramException


@dataclass(frozen=True)
class TelegramRetry:
    attempts: int = 3
    base_delay: float = 2.0
    max_delay: float = 60.0
    raise_errors: bool = True

    def delay(self, error, attempt):
        if isinstance(error, ApiTelegramException) and error.error_code == 429:
            retry_after = getattr(error, "retry_after", None)
            if retry_after is not None:
                return min(float(retry_after), self.max_delay)
        return min(self.base_delay * (2**attempt), self.max_delay)


def is_retryable_telegram_error(error):
    if isinstance(error, ApiTelegramException):
        return error.error_code == 429 or 500 <= error.error_code <= 599
    if isinstance(error, requests_exceptions.RequestException):
        return True
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return not isinstance(error, (FileNotFoundError, PermissionError))
    return False


def robust_telegram_call(
    func,
    logger,
    max_retries=3,
    base_delay=2,
    max_delay=60,
    stop_event=None,
    raise_errors=True,
):
    policy = TelegramRetry(max_retries, base_delay, max_delay, raise_errors)
    return telegram_call(func, logger, policy, stop_event)


def telegram_call(func, logger, policy, stop_event=None):
    for attempt in range(policy.attempts):
        if stop_event and stop_event.is_set():
            return None
        try:
            return func()
        except Exception as error:
            last_try = attempt == policy.attempts - 1
            if not is_retryable_telegram_error(error) or last_try:
                if policy.raise_errors:
                    raise
                logger.warning(f"Telegram call failed: {error}")
                return None

            delay = policy.delay(error, attempt)
            _log_retry(logger, error, delay)
            if stop_event.wait(delay) if stop_event else _sleep(delay):
                return None
    return None


def _sleep(delay):
    time.sleep(delay)
    return False


def _log_retry(logger, error, delay):
    if isinstance(error, ApiTelegramException):
        logger.warning(
            f"Telegram API error {error.error_code}: {error}. "
            f"Retrying in {delay:.1f}s"
        )
    else:
        logger.warning(f"Telegram network error: {error}. Retrying in {delay:.1f}s")
