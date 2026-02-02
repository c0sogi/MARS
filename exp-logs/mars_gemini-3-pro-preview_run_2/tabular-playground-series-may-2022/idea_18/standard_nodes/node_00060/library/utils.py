import os
import random
import shutil
import numpy as np
import torch


def seed_everything(seed: int):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, checkpoint_dir):
    """
    Saves the training checkpoint. If the current checkpoint represents
    the best performance, it creates a copy named 'best_model.pth'.

    Args:
        state (dict): Dictionary containing model state_dict, optimizer state, epoch, etc.
        is_best (bool): Boolean indicating if this is the best model so far.
        checkpoint_dir (str): Directory where the checkpoint should be saved.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the latest checkpoint
    filename = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    torch.save(state, filename)

    # If this is the best model, copy it to a dedicated file
    if is_best:
        best_filename = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filename, best_filename)


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Loads a checkpoint from the specified path.

    Args:
        checkpoint_path (str): Path to the checkpoint file (.pth).
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): The device to map the tensors to ('cpu' or 'cuda').

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    # Handle potential DataParallel wrapping (keys starting with 'module.')
    state_dict = checkpoint["state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    # Load optimizer state if provided and available
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
