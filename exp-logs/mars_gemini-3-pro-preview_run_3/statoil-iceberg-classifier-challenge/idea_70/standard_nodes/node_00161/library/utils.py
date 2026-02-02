import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set python hash seed for dictionary ordering consistency
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(state, is_best, fold, checkpoint_dir=Config.CHECKPOINT_DIR):
    """
    Saves the model checkpoint.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        fold (int): The current fold index.
        checkpoint_dir (str): Directory to save the checkpoint. Defaults to Config.CHECKPOINT_DIR.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    filename = f"checkpoint_fold_{fold}.pth"
    filepath = os.path.join(checkpoint_dir, filename)

    torch.save(state, filepath)

    if is_best:
        best_filename = f"model_best_fold_{fold}.pth"
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    model, fold, optimizer=None, load_best=False, checkpoint_dir=Config.CHECKPOINT_DIR
):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        fold (int): The fold index to load.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into. Defaults to None.
        load_best (bool): If True, loads the best model for the fold. Otherwise loads the latest checkpoint.
        checkpoint_dir (str): Directory containing the checkpoint. Defaults to Config.CHECKPOINT_DIR.

    Returns:
        dict: The loaded checkpoint dictionary (containing epoch, best_score, etc.).

    Raises:
        FileNotFoundError: If the specified checkpoint file does not exist.
    """
    if load_best:
        filename = f"model_best_fold_{fold}.pth"
    else:
        filename = f"checkpoint_fold_{fold}.pth"

    filepath = os.path.join(checkpoint_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    # Load to CPU first to avoid GPU OOM if multiple models are loaded,
    # or let torch handle device mapping if model is already on GPU.
    # Here we rely on default behavior but ensure map_location matches device if needed.
    device = next(model.parameters()).device
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
