import os
import shutil
import torch
import numpy as np
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


def compute_levenshtein(preds, targets):
    """
    Computes the mean Levenshtein distance between a list of predictions and targets.

    Args:
        preds (list of str): List of predicted InChI strings.
        targets (list of str): List of ground truth InChI strings.

    Returns:
        float: The mean Levenshtein distance.
    """
    distances = []
    # Iterate over pairs of predictions and targets
    for p, t in zip(preds, targets):
        # Calculate edit distance for the pair
        dist = nltk.edit_distance(p, t)
        distances.append(dist)

    # Return the mean distance
    return float(np.mean(distances))


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint to the working directory.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Flag indicating if this is the best model so far.
        filename (str): Name of the checkpoint file.
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Construct full path for the checkpoint
    filepath = os.path.join(Config.WORKING_DIR, filename)

    # Save the checkpoint
    torch.save(state, filepath)

    # If this is the best model, create a copy at the specific best model path
    if is_best:
        shutil.copyfile(filepath, Config.BEST_MODEL_PATH)


def load_checkpoint(model, optimizer=None, filename="best_model.pth"):
    """
    Loads a model checkpoint from the working directory.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        filename (str): Name or path of the checkpoint file.

    Returns:
        tuple: (start_epoch, best_metric)
    """
    # Resolve the file path
    if os.path.dirname(filename):
        filepath = filename
    else:
        filepath = os.path.join(Config.WORKING_DIR, filename)

    if not os.path.exists(filepath):
        print(f"No checkpoint found at '{filepath}'")
        return 0, float("inf")

    print(f"Loading checkpoint from '{filepath}'")
    # Load to CPU first to avoid GPU OOM if mapping is tricky
    checkpoint = torch.load(filepath, map_location="cpu")

    # Load model weights
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Extract metadata
    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", float("inf"))

    return epoch, best_metric
