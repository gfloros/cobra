import yaml
import os
import dataclasses
from rich.console import Console
from contextlib import contextmanager
import sys
import pandas as pd
import common

CONSOLE = Console()

@dataclasses.dataclass
class Train_args:
    # Path to configuration file
    config: str = './config.yaml'

def make_dir(path: str) -> None:
    """
    Creates a directory if it does not exist.
    Args:
        path (str): Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_config(path: str):
    """ Loads a configuration file from a path.

    Args:
        path (str): Path to configuration file.

    Returns:
        dict: Configuration file.
    """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return config

def log_event(message):
        CONSOLE.log(message)

def log_debug(message):
    if common.DEBUG:
        CONSOLE.log(message)

def loadEstimatorResults(path):

    return pd.read_csv(path)

@contextmanager
def stdout_redirected(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different