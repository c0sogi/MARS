import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: dict, is_best: bool, fold: int, output_dir: str = Config.IDEA_DIR
):
    """
    Saves a model checkpoint to the specified directory.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Boolean flag indicating if this is the best model so far.
        fold (int): The current cross-validation fold index.
        output_dir (str): The directory to save the checkpoint files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define filenames
    filename = os.path.join(output_dir, f"checkpoint_fold_{fold}.pth")
    best_filename = os.path.join(output_dir, f"model_best_fold_{fold}.pth")

    # Save the current state
    torch.save(state, filename)

    # If this is the best model, create a copy
    if is_best:
        shutil.copyfile(filename, best_filename)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = Config.DEVICE,
):
    """
    Loads a model checkpoint from a file.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer instance to load state into.
        device (str): The device to map the checkpoint data to (e.g., 'cuda' or 'cpu').

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    # Load checkpoint with appropriate device mapping
    checkpoint = torch.load(filepath, map_location=device)

    # Load model weights
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Fallback if the checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
