import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_score(y_true, y_pred):
    """
    Computes the Pearson correlation coefficient between true and predicted scores.

    Args:
        y_true (array-like): Ground truth scores.
        y_pred (array-like): Predicted scores.

    Returns:
        float: The Pearson correlation coefficient. Returns 0.0 if calculation fails
               (e.g., constant input resulting in NaN correlation).
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if len(y_true) < 2:
        return 0.0

    try:
        # np.corrcoef returns a covariance matrix: [[1.0, r], [r, 1.0]]
        matrix = np.corrcoef(y_true, y_pred)
        score = matrix[0, 1]

        if np.isnan(score):
            return 0.0

        return float(score)
    except Exception:
        return 0.0


def get_cpc_texts(load_cached_data=True):
    """
    Returns a dictionary mapping CPC codes (e.g., 'A47') to their full textual
    descriptions based on the section code (e.g., 'A' -> 'Human Necessities').

    Implements caching using Parquet to avoid re-scanning datasets.

    Args:
        load_cached_data (bool): If True, attempts to load the mapping from a
                                 cached parquet file. Defaults to True.

    Returns:
        dict: A dictionary where keys are context codes and values are descriptions.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "cpc_texts.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return dict(zip(df["code"], df["text"]))
        except Exception:
            # If loading fails, proceed to compute from scratch
            pass

    # 2. Compute from scratch
    # Ensure output directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    unique_contexts = set()
    files_to_scan = [Config.TRAIN_FILE, Config.VAL_FILE, Config.TEST_FILE]

    for file_path in files_to_scan:
        if os.path.exists(file_path):
            try:
                # Read only the context column for efficiency
                df_subset = pd.read_csv(file_path, usecols=["context"])
                unique_contexts.update(
                    df_subset["context"].dropna().astype(str).unique()
                )
            except Exception:
                continue

    mapping = {}
    for code in unique_contexts:
        if len(code) > 0:
            section_char = code[0].upper()
            # Map using the Config definitions
            description = Config.CPC_SECTIONS.get(section_char, "")
            mapping[code] = description
        else:
            mapping[code] = ""

    # 3. Save to cache
    try:
        df_cache = pd.DataFrame(list(mapping.items()), columns=["code", "text"])
        df_cache.to_parquet(cache_path, index=False)
    except Exception:
        # If saving fails, we still return the computed mapping
        pass

    return mapping
