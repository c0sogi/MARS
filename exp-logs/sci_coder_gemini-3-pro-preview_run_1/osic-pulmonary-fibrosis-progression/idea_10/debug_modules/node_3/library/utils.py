import os
import random
import shutil
import numpy as np
import torch


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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
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
    Computes the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or torch.Tensor): Ground truth FVC values.
        y_pred (np.array or torch.Tensor): Predicted FVC values.
        sigma (np.array or torch.Tensor): Predicted confidence (std dev).

    Returns:
        float: The average metric score (negative value, higher is better).
    """
    # Convert to numpy if inputs are torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(sigma):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are float for calculation
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    sigma = sigma.astype(np.float64)

    # 1. Clip sigma at 70 ml
    sigma_clipped = np.maximum(sigma, 70)

    # 2. Calculate absolute error and clip at 1000 ml
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, 1000)

    # 3. Compute metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)


def save_checkpoint(
    state, is_best, filename="checkpoint.pth", best_filename="best_model.pth"
):
    """
    Saves the training checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Path to save the current checkpoint.
        best_filename (str): Path to save the best model copy.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads a checkpoint into the model and optionally optimizer and scheduler.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into.
        device (str or torch.device): Device to map the checkpoint to.

    Returns:
        int: The epoch number from the checkpoint (or 0 if not found).
        float: The best metric from the checkpoint (or -inf if not found).
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint not found at {filepath}")
        return 0, -float("inf")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", -float("inf"))

    print(
        f"Loaded checkpoint '{filepath}' (epoch {start_epoch}, score {best_score:.4f})"
    )

    return start_epoch, best_score
