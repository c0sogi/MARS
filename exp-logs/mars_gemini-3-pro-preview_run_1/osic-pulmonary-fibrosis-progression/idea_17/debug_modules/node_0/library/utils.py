import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for CUDA operations.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
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


def save_checkpoint(state, is_best, filename="checkpoint.pth", checkpoint_dir=None):
    """
    Saves the model checkpoint to the specified directory.
    If is_best is True, copies the file to 'best_model.pth'.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Filename for the checkpoint.
        checkpoint_dir (str): Directory to save checkpoints. Defaults to Config.WORKING_DIR/checkpoints.
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(Config.WORKING_DIR, "checkpoints")

    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(filepath, model, optimizer=None, device=Config.DEVICE):
    """
    Loads a model checkpoint from the specified path.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to.

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch, best_score, etc.).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Handle DataParallel wrapping if necessary (remove 'module.' prefix)
    state_dict = checkpoint["state_dict"]
    # Check if the model was saved with DataParallel but we are loading to a standard model
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def compute_metric(true_fvc, pred_fvc, confidence):
    """
    Computes the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        true_fvc (np.ndarray or torch.Tensor): True FVC values.
        pred_fvc (np.ndarray or torch.Tensor): Predicted FVC values.
        confidence (np.ndarray or torch.Tensor): Predicted confidence (sigma).

    Returns:
        float: The mean metric score.
    """
    # Convert tensors to numpy if necessary for calculation
    if isinstance(true_fvc, torch.Tensor):
        true_fvc = true_fvc.detach().cpu().numpy()
    if isinstance(pred_fvc, torch.Tensor):
        pred_fvc = pred_fvc.detach().cpu().numpy()
    if isinstance(confidence, torch.Tensor):
        confidence = confidence.detach().cpu().numpy()

    # Clip confidence values at 70 ml (approximate measurement uncertainty)
    sigma_clipped = np.maximum(confidence, 70)

    # Calculate absolute error and clip at 1000 ml
    delta = np.abs(true_fvc - pred_fvc)
    delta = np.minimum(delta, 1000)

    # Calculate metric formula
    metric = -(np.sqrt(2) * delta) / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)

    return np.mean(metric)


def log_metrics(metrics, prefix=""):
    """
    Prints metrics to console with full precision.

    Args:
        metrics (dict): Dictionary of metric names and values.
        prefix (str): Optional prefix for the log message (e.g., "Epoch 1 Validation").
    """
    log_parts = []
    if prefix:
        log_parts.append(f"[{prefix}]")

    for k, v in metrics.items():
        # Print full precision without rounding
        log_parts.append(f"{k}: {v}")

    print(" ".join(log_parts))
