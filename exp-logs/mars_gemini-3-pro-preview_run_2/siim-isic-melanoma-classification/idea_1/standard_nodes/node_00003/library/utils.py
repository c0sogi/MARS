import os
import random
import shutil
import numpy as np
import torch


def seed_everything(seed: int = 42):
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

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: dict, is_best: bool, checkpoint_dir: str, filename: str = "checkpoint.pth"
):
    """
    Saves a checkpoint of the model and optimizer state.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model found so far.
        checkpoint_dir (str): The directory where the checkpoint should be saved.
        filename (str): The name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)

    # Save the current checkpoint
    torch.save(state, filepath)

    # If this is the best model, create a copy named 'model_best.pth'
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "model_best.pth")
        shutil.copyfile(filepath, best_filepath)
