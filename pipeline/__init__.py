"""
pipeline/__init__.py — BOBO KIDS STUDIO
=========================================
Logging setup for the entire pipeline.

WHY LOGURU INSTEAD OF STDLIB LOGGING?
  Python's built-in `logging` module works but requires ~20 lines of
  boilerplate setup (handlers, formatters, levels). Loguru does the
  same thing in 3 lines, produces beautiful coloured terminal output,
  and auto-rotates log files. Same API, zero config overhead.

HOW TO USE IN ANY PIPELINE MODULE:
  from pipeline import get_logger
  logger = get_logger("script_generator")
  logger.info("Script generation started")
  logger.error("Gemini API failed: {error}", error=e)
  logger.debug("Scene data: {scenes}", scenes=scenes)

LOG LEVELS (from most verbose to least):
  DEBUG   — internal variable values, for development only
  INFO    — normal progress messages ("Stage 1 complete")
  WARNING — something unexpected but recoverable
  ERROR   — something failed; stage cannot continue
  CRITICAL — whole pipeline broken
"""

import sys
from pathlib import Path
from loguru import logger as _loguru_logger

# Import cfg lazily to avoid circular import issues
def _get_log_dir() -> Path:
    from config import cfg
    return cfg.LOG_DIR


def get_logger(stage_name: str, topic_slug: str = "general"):
    """
    Returns a loguru logger configured for the given stage.

    Parameters
    ----------
    stage_name : str
        Short name for the pipeline stage (e.g., "script_generator").
        Used as the log file name prefix.
    topic_slug : str
        The topic being processed (e.g., "learn_colors").
        Included in the log filename so each run has its own file.

    Returns
    -------
    loguru.Logger
        A configured logger that writes to both the terminal (coloured)
        and a log file under logs/.
    """
    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{topic_slug}_{stage_name}.log"

    # Remove any existing handlers to avoid duplicate log entries
    # when get_logger is called multiple times in the same process
    _loguru_logger.remove()

    # ── Terminal handler ─────────────────────────────────────────────────────
    # format: time | LEVEL | stage | message
    _loguru_logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            f"<cyan>{stage_name}</cyan> | "
            "<level>{message}</level>"
        ),
        level="DEBUG",
        colorize=True,
    )

    # ── File handler ─────────────────────────────────────────────────────────
    # rotation="10 MB" means a new file starts when the current one hits 10 MB
    # retention="7 days" means log files older than 7 days are auto-deleted
    # enqueue=True makes logging thread-safe (important for async stages later)
    _loguru_logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
    )

    _loguru_logger.info(f"Logger initialised for stage='{stage_name}', topic='{topic_slug}'")
    return _loguru_logger
