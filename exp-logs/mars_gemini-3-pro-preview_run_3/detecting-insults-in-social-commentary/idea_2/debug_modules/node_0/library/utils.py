import os
import random
import codecs
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_text(text):
    """
    Decodes unicode-escaped text (e.g., "Hello\\nWorld") into proper unicode strings.
    Handles NaNs and decoding errors gracefully.
    """
    if pd.isna(text):
        return ""
    try:
        # The dataset contains double-escaped unicode sequences
        return codecs.decode(str(text), "unicode_escape")
    except Exception:
        return str(text)


def calculate_auc(y_true, y_pred):
    """
    Computes the Area Under the Receiver Operating Curve (AUC).
    """
    return roc_auc_score(y_true, y_pred)


def load_data(split, load_cached_data=True, cache_dir="./working/idea_2"):
    """
    Loads data for a specific split ('train', 'val', 'test').

    Implements a caching mechanism:
    1. Checks if a processed parquet file exists in cache_dir.
    2. If found and load_cached_data is True, loads and returns it.
    3. Otherwise, loads raw CSV from ./metadata/, cleans the text,
       saves to cache_dir as parquet, and returns the dataframe.
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{split}_decoded.parquet")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Load raw metadata
    metadata_path = os.path.join("./metadata", f"{split}.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Apply deterministic processing (text cleaning)
    if "Comment" in df.columns:
        df["Comment"] = df["Comment"].apply(clean_text)

    # Save to cache
    df.to_parquet(cache_path, index=False)

    return df


def save_submission(predictions, test_df, output_dir="./submission"):
    """
    Formats and saves the submission file.

    Args:
        predictions: Array-like of probability scores (0-1).
        test_df: The test dataframe containing 'Date' and 'Comment'.
        output_dir: Directory to save the submission.csv.
    """
    os.makedirs(output_dir, exist_ok=True)

    submission = test_df.copy()
    submission["Insult"] = predictions

    # Reorder columns to match sample format: Insult, Date, Comment
    # (Assuming these are the columns based on sample_submission_null.csv description)
    cols = ["Insult", "Date", "Comment"]
    submission = submission[cols]

    output_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(output_path, index=False)
