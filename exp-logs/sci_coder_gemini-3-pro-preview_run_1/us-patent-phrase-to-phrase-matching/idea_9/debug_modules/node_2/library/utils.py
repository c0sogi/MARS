import os
import random
import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pearson(y_true, y_pred):
    """
    Computes the Pearson correlation coefficient.

    Args:
        y_true (array-like): Ground truth scores.
        y_pred (array-like): Predicted scores.

    Returns:
        float: The Pearson correlation coefficient.
    """
    # Flatten arrays to ensure 1D
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    pearson_score, _ = pearsonr(y_true, y_pred)
    return pearson_score


def get_cpc_texts(load_cached_data: bool = True):
    """
    Parses the CPC description file to map context codes to their natural language descriptions.
    Implements caching using Parquet to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        dict: A dictionary mapping CPC codes (str) to descriptions (str).
    """
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    cache_path = os.path.join(Config.working_dir, "cpc_texts.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Convert to dictionary: code -> text
            cpc_texts = pd.Series(df.text.values, index=df.code).to_dict()
            return cpc_texts
        except Exception:
            # If load fails, proceed to process from scratch
            pass

    # 2. Process data from scratch
    cpc_texts = {}

    if os.path.exists(Config.cpc_path):
        with open(Config.cpc_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # We assume the format is "CODE DESCRIPTION" or similar.
                # We split by the first whitespace.
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    code, text = parts
                    cpc_texts[code] = text
                else:
                    # Fallback if line structure is unexpected, though unlikely given dataset norms
                    # Just skip or handle gracefully
                    continue
    else:
        # If file doesn't exist (should not happen based on task desc), return empty
        print(f"Warning: {Config.cpc_path} not found.")
        return {}

    # 3. Save to cache
    data = [{"code": k, "text": v} for k, v in cpc_texts.items()]
    df = pd.DataFrame(data)

    # Save as parquet
    df.to_parquet(cache_path, index=False)

    return cpc_texts
