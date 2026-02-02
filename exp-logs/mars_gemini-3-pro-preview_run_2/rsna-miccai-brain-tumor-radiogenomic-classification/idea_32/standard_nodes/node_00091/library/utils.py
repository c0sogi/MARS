import os
import random
import numpy as np
import torch
import shutil


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Returns the available device (CUDA or CPU).

    Returns:
        torch.device: The device to be used for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    state,
    is_best,
    checkpoint_dir,
    filename="checkpoint.pth",
    best_filename="best_model.pth",
):
    """
    Saves the training checkpoint.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the files.
        filename (str): Name of the checkpoint file.
        best_filename (str): Name of the best model file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, optimizer=None, path=None, device=None):
    """
    Loads a checkpoint into the model and optional optimizer.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        path (str): Path to the checkpoint file.
        device (torch.device, optional): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch, best_score, etc.)
    """
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")

    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)

    # Load model weights
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
