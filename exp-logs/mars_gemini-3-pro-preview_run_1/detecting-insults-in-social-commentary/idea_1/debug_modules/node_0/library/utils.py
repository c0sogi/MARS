import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and OS environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Note: PyTorch/TensorFlow seeds would be set here if those libraries were imported/used within this module,
    # but strictly following requirements we only use what's needed.


def clean_text(text):
    """
    Preprocesses the text data:
    1. Removes surrounding quotes if present.
    2. Decodes unicode-escaped sequences (e.g., \\n, \\xe2).
    3. Converts to lowercase.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Strip potential surrounding quotes from the CSV format artifacts
    text = text.strip().strip('"')

    # Decode unicode escapes (e.g. turning literal "\n" into newline, "\xe2" into bytes)
    try:
        # We encode to latin1 (or utf-8) then decode with unicode_escape to interpret the literals
        # Then encode latin1 again and decode utf-8 to get the actual characters if they were utf-8 bytes
        # However, a simple decode('unicode_escape') on the string often works for standard python strings.
        # Given the dataset description "unicode-escaped text", this pattern is robust:
        text = text.encode("utf-8").decode("unicode_escape")
    except Exception:
        # Fallback if decoding fails
        pass

    # Lowercase as per Idea description
    return text.lower()


def load_dataset(split="train", load_cached_data=True):
    """
    Loads the dataset for a specific split (train, validation, test).
    Implements deterministic processing and caching.

    Args:
        split (str): One of 'train', 'validation', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Define paths
    input_dir = "./metadata"
    cache_dir = "./working/idea_1"
    os.makedirs(cache_dir, exist_ok=True)

    filename = f"{split}.csv"
    input_path = os.path.join(input_dir, filename)
    cache_path = os.path.join(cache_dir, f"{split}_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache for {split}: {e}. Re-processing.")

    # 2. Process from scratch
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Apply preprocessing
    if "Comment" in df.columns:
        df["Comment_Clean"] = df["Comment"].apply(clean_text)
    else:
        warnings.warn(f"'Comment' column missing in {split} dataset.")

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache for {split}: {e}")

    return df


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the Receiver Operating Curve (AUC).

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The AUC score.
    """
    return roc_auc_score(y_true, y_pred)


def save_submission(predictions, test_df, output_dir="./submission"):
    """
    Saves the predictions to a submission file in the required format.

    Args:
        predictions (array-like): Predicted probabilities for the test set.
        test_df (pd.DataFrame): The original test dataframe (to keep Date/Comment columns if needed,
                                though usually we just need to match the index/order).
        output_dir (str): Directory to save the submission.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create submission dataframe based on sample_submission_null.csv format
    # The format requires: Insult, Date, Comment
    # We assume test_df preserves the order of the input test.csv

    submission = test_df.copy()

    # Ensure we use the original columns if they exist, else just the prediction
    # The prompt implies we need to fill the 'Insult' column.
    submission["Insult"] = predictions

    # If the processed dataframe has 'Comment_Clean', we should probably drop it
    # and keep the original 'Comment' for the submission file to match the sample exactly,
    # or just ensure 'Insult' is updated.
    if "Comment_Clean" in submission.columns:
        submission = submission.drop(columns=["Comment_Clean"])

    output_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
