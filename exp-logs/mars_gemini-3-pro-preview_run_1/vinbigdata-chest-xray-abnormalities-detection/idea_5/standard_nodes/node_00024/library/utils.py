import os
import random
import sys
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "Main"):
    """
    Creates and configures a logger that outputs to stdout.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if logger is already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box coordinates from (x_min, y_min, x_max, y_max)
    to (center_x, center_y, width, height).

    Args:
        x (torch.Tensor): Input tensor of shape (N, 4) or (4,).

    Returns:
        torch.Tensor: Converted tensor of shape (N, 4) or (4,).
    """
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

    # center_x = (x_min + x_max) / 2
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    # center_y = (y_min + y_max) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    # width = x_max - x_min
    y[..., 2] = x[..., 2] - x[..., 0]
    # height = y_max - y_min
    y[..., 3] = x[..., 3] - x[..., 1]

    return y


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box coordinates from (center_x, center_y, width, height)
    to (x_min, y_min, x_max, y_max).

    Args:
        x (torch.Tensor): Input tensor of shape (N, 4) or (4,).

    Returns:
        torch.Tensor: Converted tensor of shape (N, 4) or (4,).
    """
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

    half_w = x[..., 2] / 2
    half_h = x[..., 3] / 2

    # x_min = center_x - width / 2
    y[..., 0] = x[..., 0] - half_w
    # y_min = center_y - height / 2
    y[..., 1] = x[..., 1] - half_h
    # x_max = center_x + width / 2
    y[..., 2] = x[..., 0] + half_w
    # y_max = center_y + height / 2
    y[..., 3] = x[..., 1] + half_h

    return y
