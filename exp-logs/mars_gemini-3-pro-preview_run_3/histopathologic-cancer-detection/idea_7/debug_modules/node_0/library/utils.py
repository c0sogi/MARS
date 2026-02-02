import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename):
    """
    Saves the model state dictionary and other metadata to a file.

    Args:
        state (dict): Dictionary containing model_state_dict, optimizer_state_dict, etc.
        filename (str): The path where the checkpoint will be saved.
    """
    save_dir = os.path.dirname(filename)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, device=Config.DEVICE):
    """
    Loads a model checkpoint from a file.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): The device to map the loaded tensors to.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    # Load model weights
    # Check if the checkpoint is a dict with 'model_state_dict' or just the state_dict itself
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available in checkpoint
    if (
        optimizer is not None
        and isinstance(checkpoint, dict)
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def calculate_metric(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true (array-like or torch.Tensor): Ground truth binary labels.
        y_pred (array-like or torch.Tensor): Predicted probabilities for the positive class.

    Returns:
        float: The ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    # Detach and move to cpu if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check for single-class edge case (AUC is undefined if only 0s or only 1s are present)
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)
