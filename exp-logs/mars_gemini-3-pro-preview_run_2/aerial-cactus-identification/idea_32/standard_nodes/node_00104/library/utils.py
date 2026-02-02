import os
import random
import shutil
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import pandas as pd


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "model_best.pth")
        shutil.copyfile(filepath, best_path)


class AverageMeter(object):
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


def calculate_roc_auc(y_true, y_scores):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_scores (np.array): Predicted probabilities.

    Returns:
        float: ROC AUC score.
    """
    # Detach tensors if necessary and convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        # Handle cases where only one class is present in the batch
        return 0.5


def cache_data(func, cache_dir, cache_file, load_cached_data=True, **kwargs):
    """
    Generic caching mechanism for deterministic data processing.

    Args:
        func (callable): The function to compute the data if cache is missing.
        cache_dir (str): Directory to store the cache.
        cache_file (str): Filename for the cache (e.g., 'data.npy' or 'data.parquet').
        load_cached_data (bool): Whether to attempt loading from cache.
        **kwargs: Arguments to pass to `func`.

    Returns:
        The data (loaded or computed).
    """
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, cache_file)

    if load_cached_data and os.path.exists(file_path):
        try:
            if file_path.endswith(".npy"):
                data = np.load(file_path, allow_pickle=True)
                # Handle 0-d arrays (scalars/objects wrapped in numpy)
                if data.shape == ():
                    data = data.item()
                return data
            elif file_path.endswith(".parquet"):
                return pd.read_parquet(file_path)
            else:
                # Default to numpy load
                data = np.load(file_path, allow_pickle=True)
                return data
        except Exception as e:
            # If load fails, we proceed to recompute
            pass

    # Compute data
    data = func(**kwargs)

    # Save data
    try:
        if isinstance(data, pd.DataFrame):
            # Ensure filename ends with parquet if it's a dataframe, or handle appropriately
            if not file_path.endswith(".parquet"):
                # If user requested .npy for a dataframe, we might need to pickle it inside npy or change extension
                # Ideally, the user provides correct extension. We will force parquet if it's a DF and path allows.
                pass
            # If the file extension matches pandas support
            if file_path.endswith(".parquet"):
                data.to_parquet(file_path)
            elif file_path.endswith(".csv"):
                data.to_csv(file_path, index=False)
            else:
                # Fallback to numpy object save
                np.save(file_path, data)
        else:
            # Assume numpy array or generic object
            np.save(file_path, data)
    except Exception:
        # If saving fails, we just return the data without caching
        pass

    return data
