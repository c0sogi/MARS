import os
import random
import shutil
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def save_checkpoint(state, is_best, filename="checkpoint.pth", folder=Config.model_dir):
    """
    Saves the model checkpoint.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
        folder (str): Directory to save the checkpoint in.
    """
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(folder, "model_best.pth")
        shutil.copyfile(filepath, best_path)


def save_submission(
    ids,
    probabilities,
    output_path=os.path.join(Config.submission_dir, "submission.csv"),
):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (list or np.array): List of image IDs.
        probabilities (list or np.array): List of predicted probabilities for the 'dog' class.
        output_path (str): Full path where the submission CSV will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame({"id": ids, "label": probabilities})

    # Ensure IDs are integers if they are numeric
    # The sample submission shows IDs as integers (1, 2, 3...)
    if pd.api.types.is_numeric_dtype(df["id"]):
        df["id"] = df["id"].astype(int)

    # Save to CSV
    df.to_csv(output_path, index=False)
