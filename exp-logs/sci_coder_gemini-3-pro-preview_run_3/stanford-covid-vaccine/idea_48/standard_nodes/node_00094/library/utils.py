import os
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    Strictly enforces Metric Integrity by:
    1. Slicing predictions/targets to the first 'pred_len' positions (68).
    2. Filtering for the specific scored columns defined in Config.
    """

    def __init__(self):
        super().__init__()
        self.pred_len = Config.pred_len
        self.scored_indices = Config.scored_classes_indices

    def forward(self, preds, targets):
        """
        Args:
            preds: Tensor of shape (Batch, SeqLen, NumClasses)
            targets: Tensor of shape (Batch, SeqLen, NumClasses)
        Returns:
            mcrmse: Scalar Tensor
        """
        # 1. Slice to scored sequence length (e.g., 68)
        # Ensure we don't exceed dimensions if shapes are already sliced
        curr_len = preds.shape[1]
        if curr_len > self.pred_len:
            preds_sliced = preds[:, : self.pred_len, :]
            targets_sliced = targets[:, : self.pred_len, :]
        else:
            preds_sliced = preds
            targets_sliced = targets

        # 2. Select specific scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
        preds_filtered = preds_sliced[:, :, self.scored_indices]
        targets_filtered = targets_sliced[:, :, self.scored_indices]

        # 3. Compute MSE per column (averaging over Batch and Sequence dimensions)
        mse = torch.mean((preds_filtered - targets_filtered) ** 2, dim=(0, 1))

        # 4. Compute RMSE per column
        rmse = torch.sqrt(mse)

        # 5. Average RMSE across columns
        return torch.mean(rmse)


def compute_mcrmse_numpy(preds, targets):
    """
    Numpy implementation of MCRMSE for validation evaluation.

    Args:
        preds: Numpy array (N, SeqLen, NumClasses)
        targets: Numpy array (N, SeqLen, NumClasses)
    """
    pred_len = Config.pred_len
    scored_indices = Config.scored_classes_indices

    # Slice sequence
    if preds.shape[1] > pred_len:
        preds = preds[:, :pred_len, :]
        targets = targets[:, :pred_len, :]

    # Filter columns
    preds = preds[:, :, scored_indices]
    targets = targets[:, :, scored_indices]

    # Reshape to (N_samples * SeqLen, N_columns)
    preds_flat = preds.reshape(-1, preds.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])

    # MSE per column
    mse = np.mean((preds_flat - targets_flat) ** 2, axis=0)

    # RMSE per column
    rmse = np.sqrt(mse)

    # Average RMSE
    return np.mean(rmse)


def get_file_hash(file_path):
    """
    Generates an MD5 hash of a file to detect changes in source data.
    """
    if not os.path.exists(file_path):
        return None
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_cache(path, data_dict):
    """
    Saves a dictionary of numpy arrays to a compressed npz file.
    Ensures the directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **data_dict)


def load_cache(path):
    """
    Loads a compressed npz file into a dictionary of numpy arrays.
    Returns None if file does not exist or fails to load.
    """
    if not os.path.exists(path):
        return None
    try:
        # allow_pickle=True is needed if arrays contain object types,
        # though strictly numerical data is preferred.
        loaded = np.load(path, allow_pickle=True)
        return {k: loaded[k] for k in loaded.files}
    except Exception as e:
        print(f"Warning: Failed to load cache at {path}. Error: {e}")
        return None
