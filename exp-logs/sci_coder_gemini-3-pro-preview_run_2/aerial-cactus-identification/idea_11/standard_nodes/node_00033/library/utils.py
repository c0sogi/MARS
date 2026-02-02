import os
import random
import numpy as np
import torch
import pandas as pd


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
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


def get_device():
    """
    Returns the available device (CUDA or CPU).

    Returns:
        torch.device: The device to be used for computation.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, filepath):
    """
    Saves the model state dictionary and other training artifacts to the specified filepath.

    Args:
        state (dict): A dictionary containing model state, optimizer state, epoch, etc.
        filepath (str): The full path where the checkpoint will be saved.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, device=None):
    """
    Loads the model state (and optionally optimizer state) from a checkpoint file.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (torch.device, optional): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    if device is None:
        device = get_device()

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)

    # Handle different saving conventions
    if isinstance(checkpoint, dict):
        # Try to find the model state dict
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Assume the dict itself is the state dict
            model.load_state_dict(checkpoint)

        # Load optimizer state if provided and available
        if optimizer is not None:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            elif "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        # Fallback if checkpoint is not a dict (unlikely for recommended usage)
        model.load_state_dict(checkpoint)

    return checkpoint


def save_submission(ids, probabilities, output_path):
    """
    Creates and saves a submission CSV file in the required format.

    Args:
        ids (list or np.array): List of image IDs.
        probabilities (list or np.array): List of predicted probabilities for 'has_cactus'.
        output_path (str): Path to save the submission CSV.
    """
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    df = pd.DataFrame({"id": ids, "has_cactus": probabilities})

    df.to_csv(output_path, index=False)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training loops.
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
