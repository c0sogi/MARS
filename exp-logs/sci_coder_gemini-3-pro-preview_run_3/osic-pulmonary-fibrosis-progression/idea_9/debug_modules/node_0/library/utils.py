import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior where possible
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


def laplace_log_likelihood(true_fvc, pred_fvc, pred_sigma):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the competition.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        true_fvc (np.array or torch.Tensor): Ground truth FVC values.
        pred_fvc (np.array or torch.Tensor): Predicted FVC values.
        pred_sigma (np.array or torch.Tensor): Predicted confidence (sigma).

    Returns:
        float: The average metric score (negative value, higher is better).
    """
    # Convert tensors to numpy if necessary for consistent calculation
    if isinstance(true_fvc, torch.Tensor):
        true_fvc = true_fvc.detach().cpu().numpy()
    if isinstance(pred_fvc, torch.Tensor):
        pred_fvc = pred_fvc.detach().cpu().numpy()
    if isinstance(pred_sigma, torch.Tensor):
        pred_sigma = pred_sigma.detach().cpu().numpy()

    # 1. Clip the confidence (sigma) at 70 ml
    sigma_clipped = np.maximum(pred_sigma, 70)

    # 2. Calculate absolute error and threshold at 1000 ml
    delta = np.abs(true_fvc - pred_fvc)
    delta = np.minimum(delta, 1000)

    # 3. Compute the metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the mean score across the batch
    return np.mean(metric)


def save_checkpoint(
    state, is_best, checkpoint_dir=Config.CHECKPOINT_DIR, filename="checkpoint.pth"
):
    """
    Saves a model checkpoint to the working directory.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=Config.DEVICE):
    """
    Loads a model checkpoint from disk.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (optional): Scheduler to load state into.
        device (str): Device to map the location to.

    Returns:
        tuple: (start_epoch, best_score)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    # Load model weights
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)  # Fallback if only weights saved

    # Load optimizer state
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", -float("inf"))

    return start_epoch, best_score
