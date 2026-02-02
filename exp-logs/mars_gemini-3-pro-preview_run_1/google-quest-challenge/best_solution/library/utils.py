import os
import random
import shutil
import numpy as np
import torch
import pandas as pd
from scipy import stats


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearman_metric(y_true, y_pred):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values of shape (N, 30).
        y_pred (np.ndarray or torch.Tensor): Predicted values of shape (N, 30).

    Returns:
        float: The mean Spearman's correlation across all 30 columns.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    corrs = []
    # Iterate over each target column
    for col_idx in range(y_true.shape[1]):
        t = y_true[:, col_idx]
        p = y_pred[:, col_idx]

        # Spearman correlation is undefined if a variable is constant.
        # We handle this by appending 0.0 or avoiding the calculation.
        if np.std(t) == 0 or np.std(p) == 0:
            corrs.append(0.0)
        else:
            # scipy.stats.spearmanr returns (correlation, pvalue)
            # We are interested in the correlation coefficient (index 0)
            val = stats.spearmanr(t, p)[0]

            # Handle potential NaNs returned by spearmanr
            if np.isnan(val):
                corrs.append(0.0)
            else:
                corrs.append(val)

    return np.mean(corrs)


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


def save_checkpoint(
    state, is_best, checkpoint_dir="./working/checkpoints", filename="checkpoint.pth"
):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model, optimizer, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads a model checkpoint.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (optional): Scheduler to load state into.
        device (str or torch.device): Device to map the checkpoint to.

    Returns:
        tuple: (epoch, best_score) loaded from checkpoint, or (0, 0.0) if not found/keys missing.
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint not found at {filepath}")
        return 0, 0.0

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", 0.0)

    return epoch, best_score


def get_metadata(split="train"):
    """
    Loads the metadata DataFrame for a specific split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    path = f"./metadata/{split}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Metadata file for split '{split}' not found at {path}"
        )
    return pd.read_csv(path)
