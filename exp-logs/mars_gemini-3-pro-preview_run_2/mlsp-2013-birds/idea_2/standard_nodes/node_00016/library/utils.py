import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the macro-averaged ROC AUC score for multi-label classification.
    Safely handles classes that may be missing from the current batch.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels, shape (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities, shape (N, num_classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        # ROC AUC requires both positive and negative samples to be defined
        if len(np.unique(y_true[:, i])) == 2:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
            except ValueError:
                # Fallback if sklearn fails for other reasons
                pass

    if not auc_scores:
        return 0.0

    return np.mean(auc_scores)


def save_checkpoint(model, optimizer, scheduler, epoch, score, filename=None):
    """
    Saves the model checkpoint including optimizer and scheduler states.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler state.
        epoch (int): The current epoch number.
        score (float): The validation score at this checkpoint.
        filename (str, optional): Path to save the checkpoint. Defaults to Config.MODEL_SAVE_PATH.
    """
    if filename is None:
        filename = Config.MODEL_SAVE_PATH

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "score": float(score),
    }

    torch.save(checkpoint, filename)


def load_checkpoint(
    model, optimizer=None, scheduler=None, filename=None, device=Config.DEVICE
):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        filename (str, optional): Path to the checkpoint file. Defaults to Config.MODEL_SAVE_PATH.
        device (torch.device): The device to map the checkpoint to.

    Returns:
        tuple: (start_epoch, best_score)
    """
    if filename is None:
        filename = Config.MODEL_SAVE_PATH

    if not os.path.exists(filename):
        # Return default values if no checkpoint exists
        return 0, 0.0

    checkpoint = torch.load(filename, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_score = checkpoint.get("score", 0.0)

    return start_epoch, best_score
