import os
import random
import numpy as np
import torch
import library.config as config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to the value in config.py.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Determines and returns the available computing device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, epoch, metric, path):
    """
    Saves the model checkpoint including optimizer state and training metadata.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): The current training epoch.
        metric (float): The validation metric (e.g., loss or accuracy).
        path (str): The file path to save the checkpoint.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric": metric,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device=None):
    """
    Loads a model checkpoint.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (torch.device, optional): The device to map the checkpoint to.

    Returns:
        dict: The full checkpoint dictionary containing epoch and metric info.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def print_metric(name, value):
    """
    Helper to print metrics with full precision as requested.

    Args:
        name (str): Name of the metric.
        value (float): Value of the metric.
    """
    print(f"{name}: {value}")
