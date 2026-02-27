import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(service_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up logging for a given service with a rotating file handler.

    Args:
        service_name (str): The name of the service for which to set up logging.
        level (int, optional): The logging level. Defaults to logging.INFO.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(service_name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    service_handler = RotatingFileHandler(
        log_dir / f"{service_name}.log",
        maxBytes=2_000_000,
        backupCount=5,
    )
    service_handler.setFormatter(formatter)

    sys_handler = RotatingFileHandler(
        log_dir / "sys.log",
        maxBytes=5_000_000,
        backupCount=5,
    )
    sys_handler.setFormatter(formatter)

    logger.addHandler(service_handler)
    logger.addHandler(sys_handler)
    return logger
