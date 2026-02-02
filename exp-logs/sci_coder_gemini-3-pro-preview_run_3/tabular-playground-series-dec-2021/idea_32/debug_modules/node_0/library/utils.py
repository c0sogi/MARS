import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.

    Crucially, this function configures CuDNN to prioritize performance over
    strict bit-exact determinism, as per the experimental strategy (Lesson 00070).

    Args:
        seed (int): The random seed to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Disable strict determinism to allow CuDNN to select the fastest convolution algorithms.
    # This is essential for maximizing throughput on the A100 GPU.
    torch.backends.cudnn.deterministic = Config.CUDNN_DETERMINISTIC
    torch.backends.cudnn.benchmark = Config.CUDNN_BENCHMARK


def save_checkpoint(state: dict, filename: str):
    """
    Saves the training state (model, optimizer, epoch, etc.) to a file.

    Args:
        state (dict): The dictionary containing the state to save.
        filename (str): The path where the checkpoint will be saved.
    """
    # Ensure the directory exists before saving
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filename)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = Config.DEVICE,
):
    """
    Loads a checkpoint from a file into the provided model and optional optimizer.

    This function handles both raw state dictionaries and wrapped checkpoint dictionaries
    (e.g., {'model_state_dict': ..., 'optimizer_state_dict': ...}).

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): The device to map the checkpoint data to (e.g., 'cuda' or 'cpu').

    Returns:
        dict: The loaded checkpoint dictionary. This allows the caller to access
              additional metadata like 'epoch' or 'best_score'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    # Load checkpoint to the specified device
    checkpoint = torch.load(path, map_location=device)

    # 1. Load Model Weights
    # Check if the checkpoint is a wrapper dict or a direct state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Assume the checkpoint itself is the state dict
        model.load_state_dict(checkpoint)

    # 2. Load Optimizer State (if provided and available)
    if optimizer is not None and isinstance(checkpoint, dict):
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
