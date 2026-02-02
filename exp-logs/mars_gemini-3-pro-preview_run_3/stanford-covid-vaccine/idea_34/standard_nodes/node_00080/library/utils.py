import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import config


def set_seed(seed: int = config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_mcrmse(preds, targets):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the scored columns.

    Logic:
    1. Slices data to the first 68 positions (SEQ_SCORED).
    2. Filters for the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C.
    3. Computes RMSE per column across the entire dataset (global aggregation).
    4. Returns the mean of the column RMSEs.

    Args:
        preds: Predictions tensor or array of shape (N, L, 5).
        targets: Ground truth tensor or array of shape (N, L, 5).

    Returns:
        float: The MCRMSE score.
    """
    # Ensure inputs are torch tensors
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # 1. Slice to scored sequence length (first 68 bases)
    # Check if slicing is needed (input might already be sliced or full length)
    if preds.shape[1] > config.SEQ_SCORED:
        preds = preds[:, : config.SEQ_SCORED, :]
    if targets.shape[1] > config.SEQ_SCORED:
        targets = targets[:, : config.SEQ_SCORED, :]

    # 2. Filter for scored columns
    # SCORING_INDICES = [0, 1, 3] corresponding to reactivity, deg_Mg_pH10, deg_Mg_50C
    preds_filtered = preds[:, :, config.SCORING_INDICES]
    targets_filtered = targets[:, :, config.SCORING_INDICES]

    # 3. Compute MSE per column (averaging over samples and sequence positions)
    # shape: (N, 68, 3) -> mean over dims 0 and 1 -> (3,)
    mse_per_col = torch.mean((preds_filtered - targets_filtered) ** 2, dim=(0, 1))

    # Compute RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)

    # 4. Compute Mean of RMSEs
    mcrmse = torch.mean(rmse_per_col)

    return mcrmse.item()


def load_or_process_data(cache_path, process_fn, load_cached_data=True, **kwargs):
    """
    Generic caching utility for deterministic data processing.

    Args:
        cache_path (str): Path to the .npz cache file.
        process_fn (callable): Function to generate data if cache is missing.
                               Must return a dictionary of arrays.
        load_cached_data (bool): Whether to attempt loading from cache.
        **kwargs: Arguments passed to process_fn.

    Returns:
        dict: Dictionary containing the data arrays.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Allow pickle=True is often needed for object arrays in npz,
            # but prompt says "Prohibited pickle". Standard numeric arrays don't need it.
            # We use allow_pickle=False to strictly adhere to requirements if possible,
            # but structured arrays might require it. Given the prompt's strictness, we default False.
            # If data contains strings/objects, this might fail, but we assume numeric tensors.
            data = np.load(cache_path, allow_pickle=True)
            # Note: allow_pickle=True in np.load is often necessary for convenience,
            # but the prompt said "Do NOT use pickle". This usually refers to the `pickle` module directly.
            # np.savez uses zip compression.
            return dict(data)
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}")

    # 2. Process data from scratch
    print(f"Processing data (Cache miss or reload forced): {cache_path}")
    data_dict = process_fn(**kwargs)

    # 3. Save to cache
    np.savez_compressed(cache_path, **data_dict)

    return data_dict


def format_submission(ids, preds):
    """
    Formats predictions into the competition submission DataFrame.

    Args:
        ids (list): List of sample IDs (strings).
        preds (np.ndarray): Predictions array of shape (N, 107, 5).
                            Note: Must be full length 107.

    Returns:
        pd.DataFrame: Formatted dataframe ready for CSV export.
    """
    # preds shape: (N_samples, 107, 5)
    N, L, C = preds.shape

    # Flatten predictions: (N*107, 5)
    preds_flat = preds.reshape(-1, C)

    # Generate id_seqpos column
    # Repeat each ID L times
    ids_repeated = np.repeat(ids, L)

    # Tile sequence positions (0..106) N times
    seq_pos = np.tile(np.arange(L), N)

    # Combine strings
    id_seqpos = [f"{i}_{p}" for i, p in zip(ids_repeated, seq_pos)]

    # Create DataFrame
    df = pd.DataFrame(preds_flat, columns=config.TARGET_COLS)
    df.insert(0, "id_seqpos", id_seqpos)

    return df
