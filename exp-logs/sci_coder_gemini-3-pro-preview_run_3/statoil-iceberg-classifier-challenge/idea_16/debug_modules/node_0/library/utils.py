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

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(state, is_best, fold):
    """
    Saves the model checkpoint to the directory specified in Config.

    Args:
        state (dict): The state dictionary containing model_state_dict, optimizer_state_dict, etc.
        is_best (bool): Whether this checkpoint represents the best model so far (based on validation metric).
        fold (int): The current cross-validation fold index.
    """
    filename = os.path.join(Config.CHECKPOINT_DIR, f"checkpoint_fold_{fold}.pth")
    torch.save(state, filename)

    if is_best:
        best_filename = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold}.pth"
        )
        shutil.copyfile(filename, best_filename)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads a checkpoint into the model and optionally the optimizer.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch or best_score).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def count_parameters(model):
    """
    Counts the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
