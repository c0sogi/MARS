import os
import sys
import time
import random
import psutil
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_memory_usage(step_name: str = ""):
    """
    Prints the current memory usage of the process.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / 1024 / 1024
    print(f"[Memory] {step_name}: {rss_mb:.2f} MB")


class Timer:
    """
    Context manager to measure and print the execution time of a code block.
    """

    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"\n[Timer] Starting: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"[Timer] Finished: {self.name} in {elapsed:.4f} seconds")


def normalize_matrix(
    matrix: sp.spmatrix, axis: int = 1, norm: str = "l1"
) -> sp.spmatrix:
    """
    Normalizes a sparse matrix row-wise (axis=1) or column-wise (axis=0).
    Used for transition matrices to ensure probabilities sum to 1.
    """
    # sklearn.preprocessing.normalize returns a CSR matrix by default for sparse input
    return normalize(matrix, norm=norm, axis=axis)


def apk(actual, predicted, k=12):
    """
    Computes the Average Precision at k.

    Args:
        actual: list of ground truth items (order doesn't matter for set membership,
                but duplicates in ground truth are usually treated as single relevance
                set in this specific competition metric unless specified otherwise.
                Standard MAP treats relevance as binary set membership).
        predicted: list of predicted items (ordered).
        k: maximum number of predicted items to consider.

    Returns:
        float: The average precision at k.
    """
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    # Convert actual to set for O(1) lookup
    actual_set = set(actual)

    for i, p in enumerate(predicted):
        if p in actual_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def calculate_map12(
    predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> float:
    """
    Calculates the Mean Average Precision @ 12.

    Args:
        predictions_df: DataFrame with columns ['customer_id', 'prediction'].
                        'prediction' is a space-separated string of article_ids.
        ground_truth_df: DataFrame containing transaction history.
                         Must contain ['customer_id', 'article_id'].
                         This function will aggregate article_ids by customer_id.

    Returns:
        float: The MAP@12 score.
    """
    print("Calculating MAP@12...")

    # 1. Prepare Ground Truth
    # Group by customer to get list of purchased articles
    # We ensure article_ids are strings to match prediction format
    gt_df = ground_truth_df.copy()
    gt_df["article_id"] = gt_df["article_id"].astype(str)

    # Aggregate purchases per customer
    actual_series = gt_df.groupby("customer_id")["article_id"].apply(list)

    # 2. Prepare Predictions
    # Set index for fast join
    pred_df = predictions_df.set_index("customer_id")

    # 3. Join
    # We only score customers who exist in the ground truth (validation set)
    # Inner join ensures we have both actuals and predictions
    merged = pd.DataFrame({"actual": actual_series}).join(pred_df, how="inner")

    if merged.empty:
        print("Warning: No overlap between predictions and ground truth customers.")
        return 0.0

    # 4. Compute AP@12 for each customer
    # Helper to parse prediction string to list
    def compute_row_ap(row):
        actual = row["actual"]
        pred_str = row["prediction"]

        if not isinstance(pred_str, str):
            predicted = []
        else:
            predicted = pred_str.strip().split()

        return apk(actual, predicted, k=12)

    # Apply calculation
    ap_scores = merged.apply(compute_row_ap, axis=1)

    map12 = ap_scores.mean()

    print(f"Evaluated on {len(merged)} customers.")
    print(f"MAP@12: {map12:.16f}")

    return map12
