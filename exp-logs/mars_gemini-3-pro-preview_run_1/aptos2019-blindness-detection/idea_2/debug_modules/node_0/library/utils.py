import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa metric.

    Args:
        y_true: Array-like or Tensor of ground truth labels (integers 0-4).
        y_pred: Array-like or Tensor of predicted labels (integers 0-4).

    Returns:
        float: The QWK score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are integer arrays for Cohen's Kappa
    # We round predictions just in case they are passed as floats
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred).round().astype(int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def save_checkpoint(
    model, optimizer, scheduler, epoch, best_score, filename="best_model.pth"
):
    """
    Saves the model checkpoint to the working directory defined in Config.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        epoch (int): The current epoch.
        best_score (float): The best validation score achieved so far.
        filename (str): The name of the file to save.
    """
    save_dir = Config.working_dir
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_score": best_score,
    }

    torch.save(state, file_path)


def load_checkpoint(model, optimizer=None, scheduler=None, filename="best_model.pth"):
    """
    Loads a model checkpoint from the working directory.

    Args:
        model: The PyTorch model to load weights into.
        optimizer: The optimizer to load state into (optional).
        scheduler: The scheduler to load state into (optional).
        filename (str): The filename to load.

    Returns:
        tuple: (start_epoch, best_score) where start_epoch is the next epoch to train.
    """
    file_path = os.path.join(Config.working_dir, filename)

    if not os.path.exists(file_path):
        return 0, -float("inf")

    checkpoint = torch.load(file_path, map_location=Config.device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Resume from the next epoch
    start_epoch = checkpoint.get("epoch", -1) + 1
    best_score = checkpoint.get("best_score", -float("inf"))

    return start_epoch, best_score
