import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across standard libraries and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, filename: str):
    """
    Saves the training state (model, optimizer, scheduler, etc.) to a file.

    Args:
        state (dict): Dictionary containing the state to save.
        filename (str): Path to the file where the checkpoint will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    device: str = Config.DEVICE,
):
    """
    Loads a checkpoint into the model (and optionally optimizer/scheduler).

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (optional): The scheduler to load state into.
        device (str): Device to map the location to (e.g., 'cuda', 'cpu').

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    # Load model state
    # Handle case where DataParallel was used (keys start with 'module.')
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # If the model was saved without 'module.' prefix but loaded with it (or vice versa), handle it?
    # Usually standard practice is to save model.state_dict().
    # Here we assume standard saving.
    model.load_state_dict(state_dict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the Log Loss (Binary Cross Entropy) metric.

    Args:
        y_true: Array-like or Tensor of ground truth labels (0 or 1).
        y_pred: Array-like or Tensor of predicted probabilities.

    Returns:
        float: The calculated log loss.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are flat
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Calculate log loss
    # sklearn's log_loss handles clipping internally (eps=1e-15 by default)
    loss = log_loss(y_true, y_pred, labels=[0, 1])
    return loss
