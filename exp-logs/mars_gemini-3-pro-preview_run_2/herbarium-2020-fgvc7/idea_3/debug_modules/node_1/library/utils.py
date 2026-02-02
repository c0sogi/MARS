import os
import sys
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device to be used for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class Logger:
    """
    A simple logger that writes messages to both a file and the console.
    """

    def __init__(self, log_file_path):
        """
        Initialize the Logger.

        Args:
            log_file_path (str): Path to the log file.
        """
        self.log_file_path = log_file_path

        # Ensure the directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Clear the file if it exists or create a new one
        with open(self.log_file_path, "w") as f:
            pass

    def log(self, message):
        """
        Log a message to both the file and stdout.

        Args:
            message (str): The message to log.
        """
        # Print to console
        print(message)

        # Append to file
        with open(self.log_file_path, "a") as f:
            f.write(message + "\n")


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
