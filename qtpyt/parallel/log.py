"""Per-run log files."""
from __future__ import annotations

import logging


def get_logger(name: str, filename: str = "file.log", level: int = logging.DEBUG, mode: str = "w", open_log: bool = True) -> logging.Logger:
    """Logger that writes time-stamped messages to a file.

    Parameters
    ----------
    name : str
        Logger name.
    filename : str, default 'file.log'
    level : int, default logging.DEBUG
    mode : str, default 'w'
        File mode; ``'a'`` appends.
    open_log : bool, default True
        With ``False`` messages are discarded, for example on all MPI ranks
        but one.
    """
    log = logging.getLogger(name)
    log.setLevel(level)

    fmt = logging.Formatter("%(asctime)s - %(message)s")

    if open_log:
        fh = logging.FileHandler(filename, mode)
    else:
        fh = logging.NullHandler()

    fh.setLevel(level)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log

