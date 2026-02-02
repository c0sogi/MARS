import os
import random
import logging
import numpy as np
import torch
import shutil
from library.config import Config


def seed_everything(seed=42):
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


def get_logger(name=Config.PROJECT_NAME):
    """
    Creates and configures a logger that writes to both a file and the console.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # Create formatters
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File Handler
        log_file = os.path.join(Config.WORKING_DIR, "train.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def compute_metric(true_fvc, pred_fvc, pred_sigma):
    """
    Computes the Modified Laplace Log Likelihood metric.

    Metric formula:
    sigma_clipped = max(sigma, 70)
    delta = min(|true - pred|, 1000)
    metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        true_fvc (np.array or torch.Tensor): Ground truth FVC values.
        pred_fvc (np.array or torch.Tensor): Predicted FVC values.
        pred_sigma (np.array or torch.Tensor): Predicted Confidence (sigma) values.

    Returns:
        float: The mean metric score.
    """
    # Convert to numpy if tensors
    if isinstance(true_fvc, torch.Tensor):
        true_fvc = true_fvc.detach().cpu().numpy()
    if isinstance(pred_fvc, torch.Tensor):
        pred_fvc = pred_fvc.detach().cpu().numpy()
    if isinstance(pred_sigma, torch.Tensor):
        pred_sigma = pred_sigma.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    true_fvc = np.array(true_fvc, dtype=np.float64)
    pred_fvc = np.array(pred_fvc, dtype=np.float64)
    pred_sigma = np.array(pred_sigma, dtype=np.float64)

    # Constants from Config
    sigma_clip_val = Config.SIGMA_CLIP
    max_error = Config.MAX_ERROR

    # 1. Clip sigma
    sigma_clipped = np.maximum(pred_sigma, sigma_clip_val)

    # 2. Calculate absolute error and clip it
    abs_error = np.abs(true_fvc - pred_fvc)
    delta = np.minimum(abs_error, max_error)

    # 3. Compute metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Base filename for the checkpoint.
    """
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Loads a checkpoint into the model (and optionally optimizer/scheduler).

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into.

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch/score).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def count_parameters(model):
    """
    Counts the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss during training.
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
