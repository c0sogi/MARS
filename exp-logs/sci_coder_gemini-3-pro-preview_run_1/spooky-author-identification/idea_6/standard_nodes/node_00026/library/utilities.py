import os
import random
import string
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library.configuration import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_log_loss(y_true, y_pred):
    """
    Computes the multi-class logarithmic loss.

    Args:
        y_true (array-like): True labels (indices or one-hot).
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The log loss value.
    """
    # Sklearn's log_loss handles the clipping internally (eps=1e-15 by default)
    # which matches the competition requirement: max(min(p,1-10^-15),10^-15)
    return log_loss(y_true, y_pred, labels=list(range(Config.NUM_CLASSES)))


def extract_meta_features(
    df, text_col="text", cache_name="default", load_cached_data=True
):
    """
    Extracts explicit meta-features: Sentence Character Length, Word Count,
    and Punctuation Density. Implements caching to disk using Parquet.

    Args:
        df (pd.DataFrame): Input dataframe containing the text column.
        text_col (str): Name of the column containing text.
        cache_name (str): Unique identifier for the cache file (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: A DataFrame containing only the extracted meta-features.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"meta_features_{cache_name}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            meta_features = pd.read_parquet(cache_path)
            # Verify length matches input df to ensure cache validity
            if len(meta_features) == len(df):
                return meta_features
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute features from scratch
    texts = df[text_col].fillna("").astype(str)

    # Feature 1: Character Length
    char_len = texts.apply(len)

    # Feature 2: Word Count
    word_count = texts.apply(lambda x: len(x.split()))

    # Feature 3: Punctuation Density
    # Count punctuation characters and divide by character length (avoid division by zero)
    punct_chars = set(string.punctuation)
    punct_count = texts.apply(lambda x: sum(1 for char in x if char in punct_chars))
    punct_density = punct_count / char_len.replace(0, 1)  # Avoid div by zero

    meta_features = pd.DataFrame(
        {
            "meta_char_len": char_len,
            "meta_word_count": word_count,
            "meta_punct_density": punct_density,
        }
    )

    # 3. Save to cache
    try:
        meta_features.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

    return meta_features
