import os
import shutil
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


def save_checkpoint(state, is_best, fold):
    """
    Saves the model checkpoint to the working directory.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        fold (int): The current cross-validation fold.
    """
    filename = os.path.join(Config.WORKING_DIR, f"checkpoint_fold_{fold}.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(Config.WORKING_DIR, f"model_best_fold_{fold}.pth")
        shutil.copyfile(filename, best_filename)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Loads a model checkpoint.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        tuple: (start_epoch, best_score)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", float("inf"))

    return start_epoch, best_score


def save_submission(predictions, test_ids, filename=Config.SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        predictions (np.ndarray or list): Probabilities of being an iceberg.
        test_ids (np.ndarray or list): Corresponding image IDs.
        filename (str): Path to save the CSV.
    """
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame({"id": test_ids, "is_iceberg": predictions})

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
