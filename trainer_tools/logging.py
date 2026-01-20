from .utils import is_notebook
import logging, sys


def configure_logging(
    level="INFO", fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S", logger_name=None
):
    """
    Configures logging.
    If running in a notebook, ensures logs are directed to stdout to be visible in cell outputs.
    """
    numeric_level = logging._nameToLevel.get(level, level) if isinstance(level, str) else level

    handlers = None
    if is_notebook():
        handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    root = logging.getLogger()
    if logger_name:
        logging.getLogger(logger_name).setLevel(numeric_level)

    if root.handlers and is_notebook():
        handler = root.handlers[0]
        handler.flush = getattr(sys.stdout, "flush", None) or (lambda: None)

    return root
