from .utils import is_notebook
import logging, sys


def configure_logging(level="INFO", fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S", logger_name=None):
    """
    Configures logging specifically for this library.
    If running in a notebook, ensures logs are directed to stdout to be visible in cell outputs.
    """
    numeric_level = logging._nameToLevel.get(level, level) if isinstance(level, str) else level

    if logger_name is None:
        logger_name = __name__.split(".")[0]

    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout) if is_notebook() else logging.StreamHandler(sys.stderr)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    if is_notebook():
        handler.flush = getattr(sys.stdout, "flush", None) or (lambda: None)

    return logger
