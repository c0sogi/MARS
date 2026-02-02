import os
import sys
import time
import math
import torch
import numpy as np
import random
from datetime import timedelta
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the seed for reproducibility using the method defined in Config.
    """
    Config.set_seed(seed)


def get_device() -> torch.device:
    """
    Returns the appropriate torch device (CUDA if available, else CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Logger:
    """
    A simple logger that writes messages to both stdout and a file.
    """

    def __init__(self, filename: str = "train.log"):
        self.log_path = os.path.join(Config.WORKING_DIR, filename)
        # Create directory if it doesn't exist (handled in Config, but safe to check)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Initialize file (overwrite if exists to start fresh for a new run)
        with open(self.log_path, "w") as f:
            f.write(f"Log file created at {time.ctime()}\n")

    def log(self, message: str):
        """
        Prints message to console and appends to log file.
        """
        print(message)
        with open(self.log_path, "a") as f:
            f.write(str(message) + "\n")

    def log_metrics(self, metrics: dict, prefix: str = ""):
        """
        Logs a dictionary of metrics with full precision.
        """
        log_strs = []
        if prefix:
            log_strs.append(f"[{prefix}]")

        for k, v in metrics.items():
            log_strs.append(f"{k}: {v}")

        self.log(" ".join(log_strs))


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training loops.
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


def format_time(seconds):
    """
    Formats seconds into a human-readable string (HH:MM:SS).
    """
    return str(timedelta(seconds=int(round(seconds))))


def time_since(since, percent):
    """
    Calculates elapsed time and estimates remaining time based on progress percentage.
    """
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return f"{format_time(s)} (remain {format_time(rs)})"
