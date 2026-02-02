import os
import sys
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the PyTorch device based on the configuration.

    Returns:
        torch.device: The device (cpu or cuda) to be used for computation.
    """
    return torch.device(Config.DEVICE)


class Logger:
    """
    A simple logger utility that writes messages to both the console (stdout)
    and a log file in the working directory.
    """

    def __init__(self, filename: str = "training_log.txt"):
        """
        Initialize the Logger.

        Args:
            filename (str): The name of the log file to create in Config.WORKING_DIR.
        """
        self.log_file_path = os.path.join(Config.WORKING_DIR, filename)

        # Ensure the directory exists (redundant if Config.setup() is called, but safe)
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        self.file = open(self.log_file_path, "w")
        self.stdout = sys.stdout

    def log(self, message: str):
        """
        Prints the message to stdout and appends it to the log file.

        Args:
            message (str): The message string to log.
        """
        # Print to standard output
        print(message)

        # Write to file with newline
        self.file.write(message + "\n")
        self.file.flush()

    def close(self):
        """
        Closes the file handle.
        """
        if self.file:
            self.file.close()

    def __enter__(self):
        """
        Support for context manager entry.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Support for context manager exit, ensuring file closure.
        """
        self.close()
