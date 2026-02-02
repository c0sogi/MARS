import pandas as pd
import numpy as np


def apk(actual, predicted, k=12):
    """
    Computes the Average Precision at k.

    Parameters
    ----------
    actual : list or np.array
        A sequence of ground truth elements (relevant items).
    predicted : list
        A sequence of predicted elements.
    k : int, optional
        The maximum number of predicted elements to consider.

    Returns
    -------
    score : float
        The average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    # Convert actual to a set for O(1) lookup speed
    actual_set = set(actual)

    for i, p in enumerate(predicted):
        # Check if the predicted item is relevant and hasn't been predicted yet (no duplicates in prediction counting)
        if p in actual_set and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual_set:
        return 0.0

    # The denominator is the minimum of the number of relevant items and k
    return score / min(len(actual_set), k)


def mapk(actual, predicted, k=12):
    """
    Computes the Mean Average Precision at k.

    Parameters
    ----------
    actual : list of lists
        A list of ground truth sequences.
    predicted : list of lists
        A list of predicted sequences.
    k : int, optional
        The cutoff rank.

    Returns
    -------
    score : float
        The mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def calculate_map12(val_df, submission_df, k=12):
    """
    Calculates the MAP@12 score for a given validation set and submission.

    Parameters
    ----------
    val_df : pd.DataFrame
        The validation transactions. Must contain 'customer_id' and 'article_id'.
    submission_df : pd.DataFrame
        The predictions. Must contain 'customer_id' and 'prediction' (space-separated string).
    k : int, optional
        The cutoff rank (default 12).

    Returns
    -------
    map_score : float
        The calculated MAP@12 score.
    """
    # Ensure article_ids are strings for consistent comparison
    val_df = val_df.copy()
    # Fix: Pad article_ids to 10 digits to match prediction format
    val_df["article_id"] = val_df["article_id"].apply(lambda x: f"{int(x):010d}")

    # Aggregate ground truth: Get unique articles purchased by each customer
    # We use unique() because standard MAP for recommender systems typically treats
    # the set of purchased items as the target.
    ground_truth = val_df.groupby("customer_id")["article_id"].unique().reset_index()
    ground_truth.columns = ["customer_id", "actual"]

    # Prepare submission
    submission_df = submission_df.copy()
    submission_df["customer_id"] = submission_df["customer_id"].astype(str)

    # Merge ground truth with predictions
    # We perform a left merge on ground_truth because the metric is defined over
    # customers who actually made purchases in the test period.
    merged = ground_truth.merge(submission_df, on="customer_id", how="left")

    # Handle missing predictions (if any customer in val is missing from sub)
    merged["prediction"] = merged["prediction"].fillna("")

    # Convert space-separated prediction strings to lists
    merged["predicted_list"] = merged["prediction"].apply(
        lambda x: x.strip().split() if x.strip() else []
    )

    # Calculate MAP
    score = mapk(merged["actual"].tolist(), merged["predicted_list"].tolist(), k=k)

    return score
