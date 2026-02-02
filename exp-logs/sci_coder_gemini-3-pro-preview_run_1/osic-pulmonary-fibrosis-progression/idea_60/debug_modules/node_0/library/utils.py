import os
import shutil
import torch
import numpy as np
import random
from library.config import Config


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to Config.set_seed for consistency.
    """
    Config.set_seed(seed)


def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Args:
        actual_fvc (np.array or torch.Tensor): True FVC values.
        predicted_fvc (np.array or torch.Tensor): Predicted FVC values.
        confidence (np.array or torch.Tensor): Predicted confidence (sigma).

    Returns:
        float: The average metric score (negative value, higher is better).
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(actual_fvc):
        actual_fvc = actual_fvc.detach().cpu().numpy()
    if torch.is_tensor(predicted_fvc):
        predicted_fvc = predicted_fvc.detach().cpu().numpy()
    if torch.is_tensor(confidence):
        confidence = confidence.detach().cpu().numpy()

    # Ensure inputs are float for calculation
    actual_fvc = actual_fvc.astype(np.float64)
    predicted_fvc = predicted_fvc.astype(np.float64)
    confidence = confidence.astype(np.float64)

    # 1. Clip confidence (sigma) at 70 ml
    # sigma_clipped = max(sigma, 70)
    sigma_clipped = np.maximum(confidence, Config.MIN_CONFIDENCE_CLIP)

    # 2. Calculate absolute error (delta) and clip at 1000 ml
    # delta = min(|true - pred|, 1000)
    abs_error = np.abs(actual_fvc - predicted_fvc)
    delta = np.minimum(abs_error, Config.MAX_ERROR_CLIP)

    # 3. Compute Metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
    """
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(
    model, optimizer=None, filename="best_model.pth", device=Config.DEVICE
):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        filename (str): Name of the checkpoint file to load.
        device (str): Device to map the location to.

    Returns:
        int: The epoch to resume from (if found), else 0.
        float: The best metric score (if found), else -inf.
    """
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)

    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}")
        return 0, -float("inf")

    print(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", -float("inf"))

    return start_epoch, best_score


def log_message(message, log_file="train_log.txt"):
    """
    Prints a message to console and appends it to a log file.
    """
    print(message)
    log_path = os.path.join(Config.WORKING_DIR, log_file)
    with open(log_path, "a") as f:
        f.write(message + "\n")
