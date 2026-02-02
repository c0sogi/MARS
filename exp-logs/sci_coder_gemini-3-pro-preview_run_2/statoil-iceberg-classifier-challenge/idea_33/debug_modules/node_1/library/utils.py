import os
import random
import shutil
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint to the specified filename.
    If is_best is True, copies the checkpoint to 'model_best.pth' in the same directory.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): The path to save the checkpoint.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filename)

    if is_best:
        best_filename = (
            os.path.join(directory, "model_best.pth") if directory else "model_best.pth"
        )
        shutil.copyfile(filename, best_filename)


def load_checkpoint(filename, model, optimizer=None):
    """
    Loads a checkpoint from the specified filename into the model and optionally the optimizer.

    Args:
        filename (str): The path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The loaded checkpoint dictionary.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No checkpoint found at '{filename}'")

    # Load to CPU first to avoid GPU mapping issues if devices differ
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    # Load model state
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
