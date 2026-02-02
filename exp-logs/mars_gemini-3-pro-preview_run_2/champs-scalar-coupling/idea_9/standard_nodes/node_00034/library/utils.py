import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training loops.
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


def get_target_stats(df: pd.DataFrame = None, load_cached_data: bool = True):
    """
    Computes or loads the mean and standard deviation for the scalar coupling constant
    for each coupling type. Used for target normalization.

    Args:
        df (pd.DataFrame, optional): Training DataFrame containing 'type' and 'scalar_coupling_constant'.
                                     Required if cache is not found or load_cached_data is False.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        stats (dict): Dictionary mapping type index (int) to (mean, std).
    """
    cache_path = os.path.join(Config.WORKING_DIR, "target_stats.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            means = data["means"]
            stds = data["stds"]
            # Reconstruct dictionary: index -> (mean, std)
            stats = {i: (means[i], stds[i]) for i in range(len(means))}
            return stats
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute from scratch
    if df is None:
        raise ValueError(
            "DataFrame 'df' is required to compute stats when cache is missing or ignored."
        )

    stats = {}
    means = np.zeros(len(Config.COUPLING_TYPES))
    stds = np.zeros(len(Config.COUPLING_TYPES))

    # Iterate through types defined in Config to ensure consistent ordering/indexing
    for i, c_type in enumerate(Config.COUPLING_TYPES):
        subset = df[df["type"] == c_type]

        if len(subset) > 0:
            vals = subset["scalar_coupling_constant"].values
            m = np.mean(vals)
            s = np.std(vals)
        else:
            # Fallback for empty types (should not happen in this dataset)
            m = 0.0
            s = 1.0

        stats[i] = (m, s)
        means[i] = m
        stds[i] = s

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.savez(cache_path, means=means, stds=stds)

    return stats


def calculate_log_mae(
    preds: torch.Tensor, targets: torch.Tensor, types: torch.Tensor, stats: dict = None
):
    """
    Calculates the Log Mean Absolute Error (LMAE) metric, averaged across coupling types.

    Args:
        preds (torch.Tensor): Predictions tensor of shape (N,).
        targets (torch.Tensor): Ground truth tensor of shape (N,).
        types (torch.Tensor): Coupling type indices tensor of shape (N,).
        stats (dict, optional): Dictionary {type_idx: (mean, std)}.
                                If provided, it assumes preds/targets are normalized and
                                scales the error back to original units before logging.

    Returns:
        torch.Tensor: Scalar tensor containing the metric value.
    """
    # Ensure inputs are tensors
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    if not isinstance(types, torch.Tensor):
        types = torch.tensor(types)

    unique_types = torch.unique(types)
    log_maes = []

    for t in unique_types:
        t_idx = t.item()
        mask = types == t

        p_sub = preds[mask]
        t_sub = targets[mask]

        # Calculate Mean Absolute Error for this type
        mae = torch.abs(p_sub - t_sub).mean()

        # If stats provided, we are in normalized space.
        # Scale MAE to original space: MAE_real = MAE_norm * std
        if stats is not None:
            _, s = stats[t_idx]
            mae = mae * s

        # Metric is Log(MAE). We add a small epsilon for numerical stability,
        # though physically exact 0 error is unlikely.
        log_mae = torch.log(mae + 1e-9)
        log_maes.append(log_mae)

    if not log_maes:
        return torch.tensor(0.0, device=preds.device)

    # Average the log MAEs across types
    return torch.stack(log_maes).mean()


def denormalize_predictions(preds: np.ndarray, types: np.ndarray, stats: dict):
    """
    Inverse transforms normalized predictions back to their original scale.
    Used for generating the final submission file.

    Args:
        preds (np.ndarray): Normalized predictions array (N,).
        types (np.ndarray): Type indices array (N,).
        stats (dict): Dictionary {type_idx: (mean, std)}.

    Returns:
        np.ndarray: Denormalized predictions.
    """
    denorm_preds = np.zeros_like(preds)
    unique_types = np.unique(types)

    for t_idx in unique_types:
        mask = types == t_idx
        m, s = stats[t_idx]
        # Inverse transform: x = z * std + mean
        denorm_preds[mask] = preds[mask] * s + m

    return denorm_preds
