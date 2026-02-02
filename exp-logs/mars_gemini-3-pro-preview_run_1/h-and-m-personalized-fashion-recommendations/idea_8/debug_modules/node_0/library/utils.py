import pandas as pd
import numpy as np
import time
import os
import random
import torch
import gc


class Timer:
    """
    Context manager to measure execution time.
    """

    def __init__(self, name="Task"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.name}] Start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"[{self.name}] Done. Elapsed: {elapsed:.2f}s")


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across numpy, random, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def reduce_mem_usage(df, verbose=True):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Prioritize float32 over float16 for numerical stability
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            f"Mem. usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )
    return df


def apk(actual, predicted, k=12):
    """
    Computes the average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=12):
    """
    Computes the mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def calculate_map12(predictions, ground_truth, k=12):
    """
    Calculates MAP@12 given predictions and ground truth.

    Args:
        predictions (pd.DataFrame): DataFrame with columns ['customer_id', 'prediction']
                                    Prediction column should be space-separated strings of article_ids.
        ground_truth (pd.DataFrame): DataFrame with columns ['customer_id', 'article_id']
                                     Can be long format (one row per purchase) or grouped.
        k (int): Cutoff for MAP calculation.

    Returns:
        float: The MAP@12 score.
    """
    # Ensure consistent column names
    pred_df = predictions.copy()
    gt_df = ground_truth.copy()

    # If ground truth is in long format (multiple rows per customer), group it
    if "article_id" in gt_df.columns and gt_df["customer_id"].duplicated().any():
        gt_df = gt_df.groupby("customer_id")["article_id"].apply(list).reset_index()
    elif "article_id" in gt_df.columns and not isinstance(
        gt_df.iloc[0]["article_id"], list
    ):
        # Already unique but single items? convert to list
        gt_df["article_id"] = gt_df["article_id"].apply(lambda x: [x])

    # Merge
    merged = pd.merge(gt_df, pred_df, on="customer_id", how="left")

    # Fill missing predictions with empty string
    merged["prediction"] = merged["prediction"].fillna("")

    # Helper to standardize article IDs to string format (10 digits)
    def process_actual(x):
        if isinstance(x, list):
            return [
                str(i).zfill(10) if isinstance(i, (int, float)) else str(i) for i in x
            ]
        elif isinstance(x, (int, float, str)):
            return [str(x).zfill(10) if isinstance(x, (int, float)) else str(x)]
        return []

    def process_pred(x):
        if isinstance(x, str):
            return x.strip().split()
        elif isinstance(x, list):
            return [str(i) for i in x]
        return []

    actuals = merged["article_id"].apply(process_actual).tolist()
    preds = merged["prediction"].apply(process_pred).tolist()

    return mapk(actuals, preds, k=k)
