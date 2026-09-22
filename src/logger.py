import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(
        log_file: str | Path = "logs/pdf_converter.log",
        level: int = logging.INFO,
        log_http: bool = True,) -> None:
    """
    Configure the ROOT logger once: console + rotating file.
    Every named logger (ours, httpx, google_genai, ...) propagates here.
    Call once at program start, before anything else logs.
    """
    global _configured
    if _configured:
        return

    formatter = logging.Formatter(fmt=_FMT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()          # discard any stray basicConfig handler
    root.setLevel(level)
    root.addHandler(console)

    # The file handler is best-effort. On Cloud Run stdout is what Cloud Logging
    # collects, and the filesystem may be read-only or in-memory, so a failure
    # to open the log file must never stop the job from starting.
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("File logging disabled (%s: %s); logging to stdout only",
                     type(exc).__name__, exc)

    if log_http:
        # httpx emits: HTTP Request: POST <full url> "200 OK"
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("google_genai").setLevel(logging.DEBUG)

    _configured = True


def setup_logger(name: str) -> logging.Logger:
    """
    Return a named logger. No handlers are attached here — output is
    handled by the root logger set up in configure_logging().
    """
    return logging.getLogger(name)
