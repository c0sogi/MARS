import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_robust_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (ROC AUC) in a robust manner.
    It handles cases where a batch might not contain samples for all classes
    by calculating the metric per class and averaging only over classes
    present in the ground truth.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (N, C).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, C).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    num_classes = y_true.shape[1]
    class_aucs = []

    for i in range(num_classes):
        # Check if the class has both positive and negative samples
        if len(np.unique(y_true[:, i])) == 2:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            class_aucs.append(auc)
        # If a class is missing or only has one label type in this batch,
        # it is excluded from the average calculation for this specific batch.

    if len(class_aucs) == 0:
        return 0.0

    return float(np.mean(class_aucs))


def save_checkpoint(model, optimizer, epoch, score, filename):
    """
    Saves the model and optimizer state to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): Current epoch number.
        score (float): Validation score (e.g., ROC AUC).
        filename (str): Name of the file to save within the working directory.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    path = os.path.join(Config.WORKING_DIR, filename)

    state = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "score": float(score),
    }

    torch.save(state, path)


def load_checkpoint(model, optimizer, filename, device=Config.DEVICE):
    """
    Loads model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        filename (str): Name of the file to load from the working directory.
        device (torch.device): Device to map the location to.

    Returns:
        dict: The full checkpoint dictionary (containing epoch, score, etc.).
    """
    path = os.path.join(Config.WORKING_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_cache(data, filename):
    """
    Saves data to a numpy file in the working directory (caching mechanism).

    Args:
        data (np.ndarray): Data to save.
        filename (str): Filename (should end with .npy).
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    path = os.path.join(Config.WORKING_DIR, filename)
    np.save(path, data)


def load_cache(filename):
    """
    Loads data from a numpy file in the working directory.

    Args:
        filename (str): Filename to load (should end with .npy).

    Returns:
        np.ndarray or None: The loaded data, or None if file does not exist.
    """
    path = os.path.join(Config.WORKING_DIR, filename)
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None
