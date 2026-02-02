import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def laplace_log_likelihood_metric(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric as defined in the competition.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor or np.ndarray): True FVC values (ml).
        y_pred (torch.Tensor or np.ndarray): Predicted FVC values (ml).
        sigma (torch.Tensor or np.ndarray): Predicted Confidence/Std Dev (ml).

    Returns:
        float: The average metric score over the batch.
    """
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma)

    # Ensure inputs are float and on the correct device
    device = y_true.device if isinstance(y_true, torch.Tensor) else Config.DEVICE
    y_true = y_true.float().to(device)
    y_pred = y_pred.float().to(device)
    sigma = sigma.float().to(device)

    # Constants
    sigma_clip_threshold = Config.METRIC_SIGMA_CLIP
    max_error_threshold = Config.METRIC_MAX_ERROR
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device))

    # 1. Clip sigma (confidence)
    sigma_clipped = torch.clamp(sigma, min=sigma_clip_threshold)

    # 2. Calculate Delta (absolute error), clipped at 1000
    abs_diff = torch.abs(y_true - y_pred)
    delta = torch.clamp(abs_diff, max=max_error_threshold)

    # 3. Compute Metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = torch.log(sqrt_2 * sigma_clipped)

    metric = -term1 - term2

    return torch.mean(metric).item()


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the training checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, Config.BEST_MODEL_PATH)


def load_checkpoint(model, optimizer=None, scheduler=None, path=None):
    """
    Loads a checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into.
        path (str, optional): Path to the checkpoint file. Defaults to best model.

    Returns:
        tuple: (start_epoch, best_score)
    """
    if path is None:
        path = Config.BEST_MODEL_PATH

    if not os.path.exists(path):
        # Return defaults if no checkpoint found
        return 0, -float("inf")

    checkpoint = torch.load(path, map_location=Config.DEVICE)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", -float("inf"))

    return start_epoch, best_score
