import os
import random
import numpy as np
import torch
import logging
import sys
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    Creates and configures a logger that writes to both a file and stdout.

    Args:
        name: Name of the logger.
        log_dir: Directory to save the log file. If None, uses Config.LOG_DIR.
    """
    if log_dir is None:
        log_dir = Config.LOG_DIR

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # File Handler
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter("%(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    return logger


def save_checkpoint(state: dict, filename: str):
    """
    Saves the training checkpoint (model state, optimizer, etc.) to a file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = "cpu",
):
    """
    Loads a training checkpoint.

    Args:
        filename: Path to the checkpoint file.
        model: The model to load weights into.
        optimizer: The optimizer to load state into (optional).
        device: Device to map the location to.

    Returns:
        The loaded checkpoint dictionary (e.g., containing 'epoch', 'best_score').
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    # Load model state
    # Handle DataParallel wrapping if necessary (remove 'module.' prefix)
    state_dict = checkpoint["state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def calculate_weighted_log_loss(y_pred, y_true):
    """
    Calculates the weighted multi-label logarithmic loss.

    Formula: L_ij = -w_j * [y_ij * log(p_ij) + (1 - y_ij) * log(1 - p_ij)]
    Loss is averaged across all rows (samples * labels).

    Args:
        y_pred: Predicted probabilities. Shape (N, 8) or (N, 8).
                Can be numpy array or torch tensor.
        y_true: Ground truth labels. Shape (N, 8).
                Can be numpy array or torch tensor.

    Returns:
        float: The calculated weighted log loss.
    """
    # Convert to numpy if tensors
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Numerical stability
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Get weights from Config
    # Config.LOSS_WEIGHTS is a tensor: [1, 1, 1, 1, 1, 1, 1, 7]
    weights = Config.LOSS_WEIGHTS.numpy()

    # Ensure shapes match
    assert (
        y_pred.shape == y_true.shape
    ), f"Shape mismatch: {y_pred.shape} vs {y_true.shape}"
    assert y_pred.shape[1] == len(
        weights
    ), f"Expected {len(weights)} classes, got {y_pred.shape[1]}"

    # Calculate Log Loss per element
    # L = - [y * log(p) + (1-y) * log(1-p)]
    log_loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Apply weights
    # Broadcast weights [8] across batch [N, 8]
    weighted_log_loss = log_loss * weights

    # Average across all rows/elements
    # Note: The prompt says "loss is averaged across all rows".
    # Since each element in the (N, 8) matrix corresponds to a row in the submission file,
    # we take the mean of the entire weighted matrix.
    return np.mean(weighted_log_loss)
