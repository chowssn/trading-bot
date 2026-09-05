"""Centralized logging configuration for the equity system.

Call setup_logging() once at bot startup.

Log files written to equity/data/logs/ (gitignored):
  app.log         — all INFO+ logs from all modules (general activity)
  errors.log      — ERROR+ only across all modules (quick error investigation)
  security.log    — write operations, auth events (consolidated here)
  advisor.log     — all Claude API calls, thread activity, synthesis
  brief.log       — morning brief pipeline runs, section timings
  screener.log    — screener runs, filter results, quality scores
  trading.log     — future: order placement, execution (placeholder)

Rotation: security.log rotates by size (10MB, keep 30 backups) since
security events are sparse — a daily file would mostly be empty. Every
other log rotates daily at local midnight, keeping 30 days of history;
rotated files older than 2 days are gzip-compressed at startup to save
space (see compress_old_logs()).
Format: timestamp | level | module | message
"""

import gzip
import logging
import logging.handlers
import shutil
from datetime import datetime, timedelta
from pathlib import Path

LOG_DIR = Path("equity/data/logs")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_RETENTION_DAYS = 30
COMPRESS_AFTER_DAYS = 2


def setup_logging(level: str = "INFO") -> None:
    """
    Call once at startup (in equity/telegram/bot.py __main__ block,
    before Application.builder()).
    Sets up all handlers. Safe to call multiple times — idempotent.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    def rotating_handler(filename: str,
                          level: int = logging.INFO) -> logging.Handler:
        if filename == "security.log":
            # Size-based — security events are sparse, so a daily file
            # would rotate mostly empty. backupCount=30 keeps roughly the
            # same multi-week window as the time-based streams below.
            h = logging.handlers.RotatingFileHandler(
                LOG_DIR / filename,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=30,
                encoding="utf-8",
            )
        else:
            # Time-based — rotate daily at local midnight, keep 30 days.
            # Rotated files are named with the default "<file>.YYYY-MM-DD"
            # suffix; compress_old_logs() gzips them once they age out.
            h = logging.handlers.TimedRotatingFileHandler(
                LOG_DIR / filename,
                when="midnight",
                interval=1,
                backupCount=LOG_RETENTION_DAYS,
                encoding="utf-8",
                utc=False,
            )
        h.setLevel(level)
        h.setFormatter(formatter)
        return h

    # Console — INFO and above
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(formatter)
    root.addHandler(console)

    # app.log — everything INFO+ (general activity)
    root.addHandler(rotating_handler("app.log", logging.INFO))

    # errors.log — ERROR+ only (quick investigation)
    root.addHandler(rotating_handler("errors.log", logging.ERROR))

    # Subsystem loggers — route specific namespaces to dedicated files.
    # Each still propagates to root (so app.log/errors.log stay a complete
    # record) — these handlers just additionally split activity by subsystem.
    subsystems = {
        "equity.telegram": "advisor.log",    # bot + advisor activity
        "equity.brief": "brief.log",         # brief pipeline
        "equity.screener": "screener.log",   # screener runs
        "equity.portfolio": "advisor.log",   # portfolio monitor (same file as advisor)
        "security": "security.log",          # write ops + auth
        "anthropic": "advisor.log",          # claude API calls
    }

    for logger_name, filename in subsystems.items():
        logger = logging.getLogger(logger_name)
        logger.addHandler(rotating_handler(filename, logging.DEBUG))

    # httpx already reaches app.log/errors.log via propagation to root —
    # giving it its own app.log handler too would double-write every line.
    # Just raise its threshold so only warnings+ show up at all (its
    # request-per-line INFO logging is too noisy to keep).
    logging.getLogger("httpx").setLevel(logging.WARNING)

    compress_old_logs()

    logging.info("Logging configured — writing to %s", LOG_DIR.resolve())


def compress_old_logs() -> None:
    """
    Gzips rotated log files older than COMPRESS_AFTER_DAYS, and deletes
    already-gzipped ones older than LOG_RETENTION_DAYS.

    Rotated files look like: errors.log.2026-09-03
    Compressed files become: errors.log.2026-09-03.gz

    TimedRotatingFileHandler's own backupCount only recognizes its own
    plain-named rotated files — once we rename one to .gz here, the
    handler can no longer see it to enforce backupCount, so retention for
    compressed files is enforced here instead, on every startup.

    security.log is skipped: RotatingFileHandler rotates it as numbered
    backups (security.log.1, security.log.2, ...) that it renames in place
    on every rollover, and compressing one out from under it would break
    that renaming sequence. It stays uncompressed at 30 backups.

    Called once at startup from setup_logging().
    """
    compress_cutoff = datetime.now() - timedelta(days=COMPRESS_AFTER_DAYS)
    delete_cutoff = datetime.now() - timedelta(days=LOG_RETENTION_DAYS)

    for log_file in LOG_DIR.glob("*.log.*"):
        if log_file.name.startswith("security.log."):
            continue

        try:
            file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
        except FileNotFoundError:
            continue  # removed by a concurrent run since the glob listed it

        if log_file.suffix == ".gz":
            if file_mtime < delete_cutoff:
                try:
                    log_file.unlink()
                    logging.debug("Deleted expired log: %s", log_file.name)
                except Exception as e:
                    logging.warning("Could not delete %s: %s", log_file.name, e)
            continue

        if file_mtime < compress_cutoff:
            gz_path = log_file.with_suffix(log_file.suffix + ".gz")
            try:
                with open(log_file, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                log_file.unlink()  # remove uncompressed after successful gz
                logging.debug("Compressed: %s -> %s", log_file.name, gz_path.name)
            except Exception as e:
                logging.warning("Could not compress %s: %s", log_file.name, e)
