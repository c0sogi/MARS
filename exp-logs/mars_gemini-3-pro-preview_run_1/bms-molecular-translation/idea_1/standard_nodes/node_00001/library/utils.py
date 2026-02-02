import os
import shutil
import torch
import nltk
from library.config import Config


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


def compute_levenshtein(predicted_text, target_text):
    """
    Computes the Levenshtein distance between two strings using NLTK.

    Args:
        predicted_text (str): The predicted InChI string.
        target_text (str): The ground truth InChI string.

    Returns:
        int: The Levenshtein distance.
    """
    return nltk.edit_distance(predicted_text, target_text)


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model state to a file. If this is the best model so far,
    it creates a copy at Config.MODEL_PATH.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file to save in WORKING_DIR.
    """
    # Ensure the directory exists (redundant if Config handles it, but safe)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, Config.MODEL_PATH)


def load_checkpoint(filename, model, optimizer=None):
    """
    Loads a model checkpoint from the specified file.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        tuple: (epoch, best_metric) loaded from the checkpoint.
               Returns (0, float('inf')) if loading fails.
    """
    if not os.path.exists(filename):
        print(f"No checkpoint found at '{filename}'")
        return 0, float("inf")

    # Load to the configured device
    checkpoint = torch.load(filename, map_location=Config.DEVICE)

    # Load model weights
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", float("inf"))

    print(f"Loaded checkpoint '{filename}' (epoch {epoch})")
    return epoch, best_metric
