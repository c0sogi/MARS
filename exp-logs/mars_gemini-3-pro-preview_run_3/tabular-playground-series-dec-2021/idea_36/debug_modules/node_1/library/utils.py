import os
import random
import copy
import numpy as np
import torch
from library.config import SEED


def seed_everything(seed: int = SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Configuration:
    - Sets seeds for random, os.environ, numpy, and torch.
    - Configures CuDNN to 'benchmark' mode (deterministic=False) as per
      Lesson 00070 to maximize kernel performance for the Deeply-Supervised
      architecture.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Prioritize performance over strict bit-exact determinism for this complex model
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_model_copy(model: torch.nn.Module):
    """
    Creates a deep copy of the model's state dictionary.

    This function is critical for the Early Stopping protocol. It allows the
    training loop to store the 'best' model weights in memory using
    copy.deepcopy(), avoiding the I/O latency of saving to disk at every
    validation step.

    Args:
        model: The PyTorch model instance.

    Returns:
        dict: A deep copy of the model's state_dict.
    """
    if isinstance(model, torch.nn.DataParallel):
        return copy.deepcopy(model.module.state_dict())
    return copy.deepcopy(model.state_dict())


def save_checkpoint(state: dict, filepath: str):
    """
    Saves a checkpoint dictionary to the specified file path.
    Ensures the directory exists before saving.

    Args:
        state: Dictionary containing model state, optimizer state, etc.
        filepath: Destination path for the checkpoint.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module, device: torch.device = None):
    """
    Loads weights from a checkpoint file into a model.

    Args:
        filepath: Path to the checkpoint file.
        model: The PyTorch model to load weights into.
        device: The device to map the weights to (default: auto-detect).

    Returns:
        dict: The full checkpoint dictionary (useful for restoring optimizer state).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Determine if we have a full checkpoint dict or just the state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel wrapping if the target model is wrapped but checkpoint wasn't, or vice-versa
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        # If the checkpoint has 'module.' prefix but model doesn't, strip it
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    return checkpoint
