import os
import sys
import random
import shutil
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "experiment", log_file: str = None) -> logging.Logger:
    """
    Creates and configures a logger that writes to both a file and stdout.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file. If None, no file handler is added.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if function is called repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (stdout)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    """

    def __init__(self, name: str = "Metric", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        # Returns a formatted string representation
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def save_checkpoint(
    state: dict,
    is_best: bool,
    filename: str = "checkpoint.pth",
    best_filename: str = "best_model.pth",
) -> None:
    """
    Saves a checkpoint of the model state.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
        best_filename (str): Name of the best model file.
    """
    # Ensure checkpoint directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(Config.CHECKPOINT_DIR, best_filename)
        shutil.copyfile(filepath, best_filepath)


def calculate_trust_score(
    pred_log_size: torch.Tensor, true_log_size: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the Trust Score defined as the absolute error between
    the predicted log file size and the true log file size.

    This metric is used to gate the experts in the Mixture of Experts architecture.

    Args:
        pred_log_size (torch.Tensor): Predicted log file sizes.
        true_log_size (torch.Tensor): Ground truth log file sizes.

    Returns:
        torch.Tensor: Absolute errors (Trust Scores).
    """
    # Ensure tensors are flattened to handle potential shape mismatches like (N, 1) vs (N,)
    return torch.abs(pred_log_size.view(-1) - true_log_size.view(-1))


def reparameterize_repvgg(model: torch.nn.Module) -> torch.nn.Module:
    """
    Traverses the model and calls 'switch_to_deploy' on any module that supports it.
    This structurally fuses multi-branch RepVGG blocks into single-path 3x3 convolutions
    for efficient inference.

    Args:
        model (torch.nn.Module): The PyTorch model containing RepVGG blocks.

    Returns:
        torch.nn.Module: The re-parameterized model.
    """
    # Iterate through all modules (including nested ones)
    for module in model.modules():
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()

    return model
