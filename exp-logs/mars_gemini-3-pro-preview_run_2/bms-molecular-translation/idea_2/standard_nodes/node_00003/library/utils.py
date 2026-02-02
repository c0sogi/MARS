import os
import shutil
import torch
import numpy as np
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


def compute_levenshtein_distance(predictions, targets):
    """
    Computes the mean Levenshtein distance between a list of predictions and targets.

    Args:
        predictions (list[str]): List of predicted strings.
        targets (list[str]): List of ground truth strings.

    Returns:
        float: The mean Levenshtein distance.
    """
    if not predictions or not targets:
        return 0.0

    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions ({len(predictions)}) and targets ({len(targets)}) must have the same length."
        )

    distances = []
    for pred, target in zip(predictions, targets):
        # nltk.edit_distance computes the Levenshtein distance
        d = nltk.edit_distance(pred, target)
        distances.append(d)

    return np.mean(distances)


def save_checkpoint(state, is_best, filepath):
    """
    Saves the model checkpoint.

    It always saves the current state to 'last_model.pth' in the same directory as filepath.
    If is_best is True, it also copies the checkpoint to the specified filepath (e.g., 'best_model.pth').

    Args:
        state (dict): State dictionary containing model parameters, optimizer, epoch, etc.
        is_best (bool): Flag indicating if this is the best model so far.
        filepath (str): The target path for the best model checkpoint.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Define a consistent name for the latest checkpoint
    last_filepath = os.path.join(directory, "last_model.pth")

    # Save the current state
    torch.save(state, last_filepath)

    # If this is the best model, copy it to the target filepath
    if is_best:
        shutil.copyfile(last_filepath, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device=None):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (torch.device, optional): The device to map the checkpoint to.

    Returns:
        dict: The loaded checkpoint dictionary if successful, else None.
    """
    if not os.path.exists(filepath):
        print(f"[-] Checkpoint not found at '{filepath}'. Starting from scratch.")
        return None

    print(f"[+] Loading checkpoint from '{filepath}'...")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint to the specific device
    # Cite debug_lesson_2: Set weights_only=False to allow loading numpy types (like best_metric)
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Load model state
    # Handle cases where the state_dict might be nested under 'state_dict' or 'model_state_dict'
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume the checkpoint itself is the state dict
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    print(f"[+] Checkpoint loaded successfully.")
    return checkpoint
