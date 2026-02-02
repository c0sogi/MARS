import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
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


def save_checkpoint(
    model, optimizer, scheduler, epoch, metric, is_best=False, filename=None
):
    """
    Saves the model state, optimizer, scheduler, and current metric.
    """
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"

    # Ensure checkpoint directory exists
    os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)

    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metric": metric,
    }

    filepath = os.path.join(Config.CHECKPOINTS_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")
        torch.save(state, best_path)


def load_checkpoint(
    model,
    filename="best_model.pth",
    optimizer=None,
    scheduler=None,
    device=Config.DEVICE,
):
    """
    Loads the model state and optionally optimizer and scheduler.
    """
    filepath = os.path.join(Config.CHECKPOINTS_DIR, filename)
    if not os.path.exists(filepath):
        # If absolute path or relative to cwd was passed
        if os.path.exists(filename):
            filepath = filename
        else:
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint.get("epoch", 0), checkpoint.get("metric", 0.0)


def mixup_data(x, y, alpha=1.0, device=Config.DEVICE):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates loss for mixup.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_auc(y_true, y_pred):
    """
    Calculates Area Under the ROC Curve.
    Handles edge cases where only one class is present in the batch.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        # Cannot calculate AUC with only one class
        return 0.5

    return roc_auc_score(y_true, y_pred)


def save_cache(data, filename):
    """
    Saves numpy data to cache directory.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    filepath = os.path.join(Config.CACHE_DIR, filename)
    np.save(filepath, data)


def load_cache(filename):
    """
    Loads numpy data from cache directory if it exists.
    """
    filepath = os.path.join(Config.CACHE_DIR, filename)
    # Check for .npy extension if not provided
    if not os.path.exists(filepath) and not filepath.endswith(".npy"):
        filepath += ".npy"

    if os.path.exists(filepath):
        return np.load(filepath, allow_pickle=False)
    return None
