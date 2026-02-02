import os
import shutil
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
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


def save_checkpoint(state, is_best, fold):
    """
    Saves the training checkpoint.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Flag indicating if this is the best model found so far.
        fold (int): The current cross-validation fold index.
    """
    # Ensure checkpoint directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    filename = f"checkpoint_fold_{fold}.pth"
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)

    torch.save(state, filepath)

    if is_best:
        best_filename = f"model_best_fold_{fold}.pth"
        best_filepath = os.path.join(Config.CHECKPOINT_DIR, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(fold, model, optimizer=None, device=Config.DEVICE, load_best=True):
    """
    Loads a checkpoint into the model and optionally the optimizer.

    Args:
        fold (int): The fold index to load.
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): The device to map the checkpoint to.
        load_best (bool): If True, loads the best model for the fold. If False, loads the latest checkpoint.

    Returns:
        tuple: (start_epoch, best_metric)
    """
    filename = (
        f"model_best_fold_{fold}.pth" if load_best else f"checkpoint_fold_{fold}.pth"
    )
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    # Handle potential DataParallel wrapping differences
    state_dict = checkpoint["state_dict"]
    # If checkpoint has 'module.' prefix but current model doesn't (or vice versa), handling might be needed.
    # Here we assume standard loading. If keys mismatch due to 'module.', strip it.
    if list(state_dict.keys())[0].startswith("module.") and not hasattr(
        model, "module"
    ):
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", None)

    return epoch, best_metric
