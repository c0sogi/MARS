import os
import shutil
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
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


def map_at_5(predictions, ground_truth):
    """
    Calculates the Mean Average Precision @ 5 (MAP@5).

    In this specific task, each image has exactly one ground truth label.
    Therefore, MAP@5 is equivalent to the Mean Reciprocal Rank (MRR)
    calculated over the top 5 predictions.

    Args:
        predictions (list or np.ndarray): A list (or array) where each element
                                          is an iterable (e.g., list) of the
                                          top 5 predicted class labels/indices.
        ground_truth (list or np.ndarray): A list (or array) of the actual
                                           class labels/indices.

    Returns:
        float: The calculated MAP@5 score.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground_truth must have the same length.")

    score = 0.0

    for preds, target in zip(predictions, ground_truth):
        # Convert to list if it's a numpy array to ensure 'in' operator works as expected
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()

        if target in preds:
            # Get the rank (0-indexed)
            rank = preds.index(target)
            # We only care about top 5
            if rank < 5:
                score += 1.0 / (rank + 1)

    return score / len(ground_truth)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): The name of the checkpoint file. Defaults to "checkpoint.pth.tar".
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define the full path for the checkpoint
    filepath = os.path.join(Config.WORKING_DIR, filename)

    # Save the checkpoint
    torch.save(state, filepath)

    # If this is the best model, copy it to the designated best model path
    if is_best:
        shutil.copyfile(filepath, Config.MODEL_SAVE_PATH)
