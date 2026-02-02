import os
import random
import numpy as np
import torch
from library.config import DEVICE


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure fully reproducible results.

    Args:
        seed (int): The random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rmse_score(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.
    Handles both PyTorch tensors and NumPy arrays.

    Args:
        y_true: Ground truth values (Tensor or array).
        y_pred: Predicted values (Tensor or array).

    Returns:
        float: The RMSE score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate MSE then RMSE
    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse))


def save_checkpoint(state, filename):
    """
    Saves the model state to a file.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        filename (str): The path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):
    """
    Loads a model checkpoint from a file.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch or best score).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    # Load checkpoint to the configured device
    checkpoint = torch.load(filename, map_location=DEVICE)

    # Load model weights
    # strict=True ensures that the keys match exactly
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # Load optimizer state if provided and present in checkpoint
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
