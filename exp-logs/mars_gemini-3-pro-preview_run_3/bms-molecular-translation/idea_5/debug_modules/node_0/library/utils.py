import os
import shutil
import torch
import nltk
from library.config import Config


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
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


def levenshtein_distance(pred: str, target: str) -> int:
    """
    Computes the Levenshtein distance between two strings using NLTK.

    Args:
        pred (str): The predicted string.
        target (str): The ground truth string.

    Returns:
        int: The edit distance.
    """
    return nltk.edit_distance(pred, target)


def save_checkpoint(state: dict, is_best: bool, filename: str = "checkpoint.pth"):
    """
    Saves the model checkpoint to the working directory.
    If is_best is True, copies the file to the best checkpoint path defined in Config.

    Args:
        state (dict): The state dictionary to save (model weights, optimizer, etc.).
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): The name of the checkpoint file.
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define the full path for the checkpoint
    filepath = os.path.join(Config.WORKING_DIR, filename)

    # Save the state dictionary
    torch.save(state, filepath)

    # If this is the best model, copy it to the designated best model path
    if is_best:
        shutil.copyfile(filepath, Config.CHECKPOINT_PATH)
        print(f"Saved best model to {Config.CHECKPOINT_PATH}")
