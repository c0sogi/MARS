import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(state, filename):
    """
    Saves the training checkpoint (model state, optimizer state, etc.) to a file.

    Args:
        state (dict): A dictionary containing the model state, optimizer state, and other metadata.
        filename (str): The path where the checkpoint file will be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Loads a checkpoint into the model and optionally the optimizer.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer instance to load state into.
        device (str): The device to map the loaded tensors to ('cpu' or 'cuda').

    Returns:
        dict: The loaded checkpoint dictionary.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load the checkpoint dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state dict
    # Handle cases where checkpoint is the state_dict itself or a dict containing it
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state dict if provided and available
    if (
        optimizer is not None
        and isinstance(checkpoint, dict)
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
