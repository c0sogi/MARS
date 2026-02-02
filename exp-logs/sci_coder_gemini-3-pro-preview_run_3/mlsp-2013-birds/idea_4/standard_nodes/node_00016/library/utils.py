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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Macro-Averaged ROC AUC score.
    Handles cases where specific classes may not be present in the ground truth
    by calculating AUC per class and averaging only valid scores.

    Args:
        y_true (np.array or torch.Tensor): Ground truth labels (N, NumClasses).
        y_pred (np.array or torch.Tensor): Predicted probabilities (N, NumClasses).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        # Only calculate AUC if the class exists in the targets
        if len(np.unique(y_true[:, i])) > 1:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
            except ValueError:
                # Fallback for extreme edge cases
                pass

    if not auc_scores:
        return 0.0

    return np.mean(auc_scores)


def save_checkpoint(state, filename):
    """
    Saves the model checkpoint to the configured checkpoint directory.

    Args:
        state (dict): State dictionary containing model, optimizer, etc.
        filename (str): Name of the file to save (e.g., 'fold_0_best.pth').
    """
    # Ensure the directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(state, filepath)


def load_checkpoint(
    filename, model, optimizer=None, scheduler=None, device=Config.DEVICE
):
    """
    Loads a model checkpoint into the provided model/optimizer/scheduler.

    Args:
        filename (str): Name of the checkpoint file to load.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into.
        device (str): Device to map the storage to.

    Returns:
        dict: The loaded checkpoint dictionary, or None if file not found.
    """
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)

    if not os.path.exists(filepath):
        # Try treating filename as a full path
        if os.path.exists(filename):
            filepath = filename
        else:
            print(f"Checkpoint not found at {filepath}")
            return None

    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume the checkpoint is the state dict itself
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None:
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        elif "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
