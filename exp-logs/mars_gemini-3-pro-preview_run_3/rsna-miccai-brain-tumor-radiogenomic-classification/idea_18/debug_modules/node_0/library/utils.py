import os
import random
import numpy as np
import torch
import datetime
import sys


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def get_device() -> torch.device:
    """
    Determines and returns the available compute device (GPU or CPU).

    Returns:
        torch.device: The device object (cuda or cpu).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Device selected: {device} ({device_name})")
    else:
        device = torch.device("cpu")
        print(f"Device selected: {device}")
    return device


class Logger:
    """
    A simple logger to track execution progress with timestamps.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the logger.

        Args:
            verbose (bool): If True, messages are printed to stdout.
        """
        self.verbose = verbose

    def log(self, message: str) -> None:
        """
        Logs a message with the current timestamp.

        Args:
            message (str): The message to log.
        """
        if self.verbose:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")
            sys.stdout.flush()

    def section(self, title: str) -> None:
        """
        Logs a section header.

        Args:
            title (str): The title of the section.
        """
        if self.verbose:
            print("\n" + "=" * 60)
            self.log(f"STARTING: {title}")
            print("=" * 60)
