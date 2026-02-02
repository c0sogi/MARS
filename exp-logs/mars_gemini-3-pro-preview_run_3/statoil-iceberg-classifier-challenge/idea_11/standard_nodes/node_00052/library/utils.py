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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(state, is_best, fold_idx, checkpoint_dir=Config.CHECKPOINT_DIR):
    """
    Saves the model checkpoint to the specified directory.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, epoch, etc.
        is_best (bool): Boolean flag indicating if this is the best model so far (based on validation metric).
        fold_idx (int): The current fold index (used for naming).
        checkpoint_dir (str): Directory where checkpoints are saved. Defaults to Config.CHECKPOINT_DIR.
    """
    # Ensure directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define filenames
    filename = f"checkpoint_fold_{fold_idx}.pth"
    filepath = os.path.join(checkpoint_dir, filename)

    # Save the current state
    torch.save(state, filepath)

    # If it's the best model, create a copy with a specific name
    if is_best:
        best_filename = f"model_best_fold_{fold_idx}.pth"
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads a model checkpoint from a file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer instance to load state into.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch, best_score, etc.).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    # Load checkpoint to the configured device
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)

    # Load model weights
    # strict=True ensures that the keys match exactly
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # Load optimizer state if provided and available in checkpoint
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
