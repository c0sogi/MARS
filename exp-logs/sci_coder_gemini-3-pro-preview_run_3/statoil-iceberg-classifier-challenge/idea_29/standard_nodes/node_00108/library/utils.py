import os
import shutil
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
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


def save_checkpoint(state, is_best, fold, checkpoint_dir):
    """
    Saves the model checkpoint to the specified directory.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether the current checkpoint represents the best model so far.
        fold (int): The current cross-validation fold index.
        checkpoint_dir (str): The directory where checkpoints should be saved.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define filenames
    filename = os.path.join(checkpoint_dir, f"checkpoint_fold_{fold}.pth")
    best_filename = os.path.join(checkpoint_dir, f"model_best_fold_{fold}.pth")

    # Save the current state
    torch.save(state, filename)

    # If this is the best model, copy it to the best_filename
    if is_best:
        shutil.copyfile(filename, best_filename)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads a model checkpoint from the specified path.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    # Load on CPU first to avoid GPU OOM if multiple models are loaded,
    # or let torch.load handle device mapping if needed.
    # Here we default to loading to the device the tensor was saved from,
    # but usually it's safer to map to the current device in the training loop.
    # For simplicity, we use default behavior.
    checkpoint = torch.load(checkpoint_path)

    # Load model weights
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Fallback if the file is just the state dict
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
