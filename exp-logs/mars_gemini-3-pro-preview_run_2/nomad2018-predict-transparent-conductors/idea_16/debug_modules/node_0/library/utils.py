import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(state: dict, filepath: str):
    """
    Saves the model checkpoint to the specified filepath.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        filepath (str): Path to save the checkpoint file.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = "cpu",
):
    """
    Loads a model checkpoint from the specified filepath.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch, loss, etc.).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def compute_rmsle(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        preds (np.ndarray or torch.Tensor): Predicted values.
        targets (np.ndarray or torch.Tensor): Ground truth values.

    Returns:
        float: The mean RMSLE across all target columns.
    """
    # Ensure inputs are numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Clip predictions to be non-negative as log is undefined for negative numbers
    # Formation energy and bandgap should physically be non-negative (or close to 0)
    preds = np.maximum(preds, 0)

    # Calculate squared logarithmic errors
    # log1p(x) = log(x + 1)
    squared_log_errors = (np.log1p(preds) - np.log1p(targets)) ** 2

    # Mean squared logarithmic error for each column
    msle_per_column = np.mean(squared_log_errors, axis=0)

    # Root mean squared logarithmic error for each column
    rmsle_per_column = np.sqrt(msle_per_column)

    # Average across columns
    mean_rmsle = np.mean(rmsle_per_column)

    return float(mean_rmsle)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
