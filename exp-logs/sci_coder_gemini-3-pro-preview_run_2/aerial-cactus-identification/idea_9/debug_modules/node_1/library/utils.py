import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(state: dict, filename: str):
    """
    Saves the model checkpoint to the working directory.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        filename (str): The name of the file to save (e.g., 'model_seed_0.pth').
    """
    # Ensure the working directory exists (Config.setup() does this, but safety first)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filepath = Config.get_cache_path(filename)
    torch.save(state, filepath)


def load_checkpoint(model, filename: str, optimizer=None, device=None):
    """
    Loads a model checkpoint from the working directory.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): The filename of the checkpoint.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str or torch.device, optional): The device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    filepath = Config.get_cache_path(filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    if device is None:
        device = Config.DEVICE

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Fallback if the checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def compute_roc_auc(y_true, y_pred):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores (probability estimates).

    Returns:
        float: The ROC AUC score.
    """
    # Detach from graph if tensors are passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)
