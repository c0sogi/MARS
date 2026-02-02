import pandas as pd
import numpy as np
import gc
import time
import contextlib


def memory_cleanup():
    """
    Aggressively releases memory by invoking the garbage collector.
    Useful after deleting large dataframes or arrays to free up RAM.
    """
    gc.collect()


class Timer(contextlib.ContextDecorator):
    """
    Context manager to track and print the runtime of specific code blocks.

    Usage:
        with Timer("Data Loading"):
            load_data()
    """

    def __init__(self, name="Task"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.name}] Starting...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"[{self.name}] Completed in {elapsed:.4f} seconds.")


def calculate_map12(val_df, submission_df):
    """
    Calculates the Mean Average Precision @ 12 (MAP@12) for the provided validation data
    and submission predictions.

    The metric is calculated as:
    MAP@12 = (1/U) * sum_{u=1}^U (1/min(m, 12)) * sum_{k=1}^min(n, 12) P(k) * rel(k)

    where:
    - U is the number of customers in the validation set.
    - m is the number of ground truth items for a customer.
    - n is the number of predictions (up to 12).
    - P(k) is the precision at cutoff k.
    - rel(k) is 1 if the item at rank k is relevant, 0 otherwise.

    Args:
        val_df (pd.DataFrame): DataFrame containing ground truth data.
                               Must have 'customer_id' and 'article_id' columns.
        submission_df (pd.DataFrame): DataFrame containing predictions.
                                      Must have 'customer_id' and 'prediction' columns.
                                      'prediction' should be a space-separated string of article_ids.

    Returns:
        float: The MAP@12 score.
    """
    print("Calculating MAP@12...")

    # Ensure consistent data types for merging
    # We work with copies to avoid modifying the originals
    val_gt = val_df[["customer_id", "article_id"]].copy()
    sub_pred = submission_df[["customer_id", "prediction"]].copy()

    val_gt["customer_id"] = val_gt["customer_id"].astype(str)
    val_gt["article_id"] = val_gt["article_id"].astype(str)
    sub_pred["customer_id"] = sub_pred["customer_id"].astype(str)

    # Group ground truth by customer to get the set of purchased items
    # We treat relevance as the set of unique items purchased in the validation period
    ground_truth = val_gt.groupby("customer_id")["article_id"].apply(set).reset_index()
    ground_truth.columns = ["customer_id", "actual_set"]

    # Merge predictions with ground truth.
    # We use a 'right' join to ensure we score ALL customers present in the validation set (ground truth),
    # as per the requirement: "All customers who made purchases during the test period are scored".
    merged = sub_pred.merge(ground_truth, on="customer_id", how="right")

    # Fill missing predictions with empty strings
    merged["prediction"] = merged["prediction"].fillna("")

    # Extract lists for iteration
    preds_list = merged["prediction"].tolist()
    actual_list = merged["actual_set"].tolist()

    scores = []

    for p_str, act_set in zip(preds_list, actual_list):
        # If no ground truth items, the term is technically undefined or 0 contribution
        # depending on interpretation, but usually filtered out or m=0 implies score 0.
        # With a right join on val_df, act_set should not be null, but could be empty if data is weird.
        if not isinstance(act_set, set) or not act_set:
            scores.append(0.0)
            continue

        m = len(act_set)
        if m == 0:
            scores.append(0.0)
            continue

        # Parse predictions (space separated string) -> list of strings
        # We take the top 12 predictions
        preds = p_str.strip().split()[:12]

        running_score = 0.0
        hits = 0

        for k, pred_item in enumerate(preds):
            # Check relevance
            if pred_item in act_set:
                hits += 1
                # Precision at k = hits / k (1-based index)
                running_score += hits / (k + 1)

        # Average Precision for this user
        ap = running_score / min(m, 12)
        scores.append(ap)

    # Mean Average Precision
    final_map = np.mean(scores)

    # Print full precision as requested
    print(f"MAP@12: {final_map:.16f}")

    return final_map
