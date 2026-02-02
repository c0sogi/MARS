import os
import random
import numpy as np
import torch
import nltk


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and other metrics during training.
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


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves the model state to a file.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        filename (str): The path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Loads the model state from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): The device to map the location to (e.g., 'cpu' or 'cuda').

    Returns:
        dict: The full checkpoint dictionary loaded from the file.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    print(f"=> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Fallback if the checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def compute_levenshtein(predictions, targets):
    """
    Computes the mean Levenshtein distance between a list of predictions and targets.

    Args:
        predictions (list of str): The predicted InChI strings.
        targets (list of str): The ground truth InChI strings.

    Returns:
        float: The mean Levenshtein distance.
    """
    if not predictions or not targets:
        return 0.0

    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")

    distances = []
    for pred, target in zip(predictions, targets):
        # nltk.edit_distance calculates Levenshtein distance
        d = nltk.edit_distance(pred, target)
        distances.append(d)

    return np.mean(distances)
