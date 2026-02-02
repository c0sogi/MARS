import os
import shutil
import torch
import nltk
import numpy as np
from library.config import Config


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


def calc_levenshtein(predictions, targets):
    """
    Calculates the mean Levenshtein distance between predictions and targets.

    Args:
        predictions (List[str]): List of predicted InChI strings.
        targets (List[str]): List of ground truth InChI strings.

    Returns:
        float: Mean Levenshtein distance over the batch.
    """
    if not predictions:
        return 0.0

    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions length ({len(predictions)}) and targets length ({len(targets)}) must match."
        )

    distances = []
    for pred, target in zip(predictions, targets):
        # nltk.edit_distance computes the number of edits required to transform one string to another
        dist = nltk.edit_distance(pred, target)
        distances.append(dist)

    return np.mean(distances)


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint to the working directory.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, epoch, etc.
        is_best (bool): If True, copies this checkpoint to the 'best_model.pth' location.
        filename (str): Name of the checkpoint file.
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, Config.MODEL_PATH)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Loads a model checkpoint from the specified file.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (object, optional): Scheduler to load state into.

    Returns:
        dict: The loaded checkpoint dictionary if successful, else None.
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint file not found at: {filepath}")
        return None

    print(f"Loading checkpoint from: {filepath}")

    # Load onto the configured device (cpu or cuda)
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)

    # Load model state dict
    state_dict = checkpoint["state_dict"]

    # Handle case where model was trained with DataParallel (keys start with 'module.')
    # but is being loaded into a standard model instance
    if list(state_dict.keys())[0].startswith("module."):
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint.get("epoch", "Unknown")
    best_score = checkpoint.get("best_score", "Unknown")
    print(f"Checkpoint loaded. Epoch: {epoch}, Best Score: {best_score}")

    return checkpoint
