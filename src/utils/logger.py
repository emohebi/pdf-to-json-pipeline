"""
Logging configuration.

Outputs to BOTH:
  - Terminal (stdout) — so you see logs in real time
  - Log file (LOGS_DIR / pipeline.log) — so logs persist after the run

Architecture:
  A single FileHandler is attached to the ROOT logger on first import.
  Each module calls setup_logger("name") which returns a child logger.
  Child loggers propagate to root, so they automatically write to both
  the terminal StreamHandler and the file FileHandler.
"""
import logging
import sys
import os

_initialized = False
_log_file_path = None


def _init_root_logger():
    """
    Attach StreamHandler + FileHandler to the root logger.
    Called once on first setup_logger() call.
    """
    global _initialized, _log_file_path

    if _initialized:
        return

    root = logging.getLogger()

    # Avoid duplicates if something else already configured root
    # We check by type to be safe
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    has_file = any(isinstance(h, logging.FileHandler) for h in root.handlers)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # --- Terminal handler ---
    if not has_stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        root.addHandler(stream_handler)

    # --- File handler ---
    if not has_file:
        log_file = _resolve_log_file_path()
        if log_file:
            try:
                # Ensure directory exists
                log_dir = os.path.dirname(log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)

                file_handler = logging.FileHandler(
                    log_file, mode="a", encoding="utf-8"
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.DEBUG)  # Capture everything in the file
                root.addHandler(file_handler)
                _log_file_path = log_file
            except Exception as e:
                # If we can't create the file handler, warn but don't crash
                print(
                    f"WARNING: Could not create log file handler "
                    f"at '{log_file}': {e}",
                    file=sys.stderr,
                )

    # Root level should be the most permissive; individual loggers
    # and handlers control their own levels.
    root.setLevel(logging.DEBUG)

    _initialized = True


def _resolve_log_file_path() -> str:
    """
    Resolve the log file path from settings.

    We import settings lazily to avoid circular imports (settings
    imports config_loader which may import logger).
    If settings aren't available yet (e.g. config.json not found),
    fall back to a sensible default.
    """
    try:
        from config.settings import PIPELINE_LOG_FILE
        return str(PIPELINE_LOG_FILE)
    except Exception:
        # Settings not available — try environment variable
        log_dir = os.getenv("LOG_DIR")
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            return os.path.join(log_dir, "pipeline.log")

        # Last resort: current working directory
        fallback = os.path.join(os.getcwd(), "output", "logs")
        os.makedirs(fallback, exist_ok=True)
        return os.path.join(fallback, "pipeline.log")


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Get or create a named logger.

    On the first call, initialises the root logger with both a
    StreamHandler (terminal) and a FileHandler (log file).
    Subsequent calls just return the named child logger.

    Args:
        name: Logger name (typically the module name).
        level: Logging level for this specific logger.

    Returns:
        A configured Logger instance.
    """
    _init_root_logger()

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Do NOT add handlers to child loggers — they propagate to root.
    # This prevents duplicate log lines.
    # (Clear any handlers that may have been added by previous code)
    if logger.handlers:
        logger.handlers.clear()
    logger.propagate = True

    return logger


def get_log_file_path() -> str:
    """Return the path to the current log file, or empty string if none."""
    return _log_file_path or ""