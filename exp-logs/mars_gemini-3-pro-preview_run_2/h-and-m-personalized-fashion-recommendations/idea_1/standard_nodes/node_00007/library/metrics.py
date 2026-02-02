import pandas as pd
import numpy as np


def apk(actual, predicted, k=12):
    """
    Computes the average precision at k.

    Parameters
    ----------
    actual : list or np.array
        A collection of elements that are to be predicted (ground truth).
    predicted : list or np.array
        A list of predicted elements (order matters).
    k : int, optional
        The maximum number of predicted elements.

    Returns
    -------
    score : float
        The average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if len(actual) == 0:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=12):
    """
    Computes the mean average precision at k.

    Parameters
    ----------
    actual : list
        A list of lists (or arrays) of ground truth elements.
    predicted : list
        A list of lists (or arrays) of predicted elements.
    k : int, optional
        The maximum number of predicted elements.

    Returns
    -------
    score : float
        The mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def calculate_map12(valid_df, sub_df, k=12):
    """
    Calculates the Mean Average Precision @ 12 (MAP@12) for the H&M dataset.

    Parameters
    ----------
    valid_df : pd.DataFrame
        DataFrame containing the ground truth transactions.
        Must contain columns 'customer_id' and 'article_id'.
    sub_df : pd.DataFrame
        DataFrame containing the submission predictions.
        Must contain columns 'customer_id' and 'prediction'.
        'prediction' should be a space-separated string of article_ids.
    k : int, optional
        The cutoff for the calculation (default is 12).

    Returns
    -------
    map_score : float
        The calculated MAP@12 score.
    """
    # Work on copies to prevent side effects
    valid = valid_df.copy()
    sub = sub_df.copy()

    # Ensure article_id is string
    valid["article_id"] = valid["article_id"].astype(str)

    # Group ground truth by customer_id to get unique items purchased
    # Using unique() ensures we treat the ground truth as a set of relevant items
    valid_grouped = valid.groupby("customer_id")["article_id"].unique().reset_index()
    valid_grouped.columns = ["customer_id", "actual"]

    # Parse predictions: split space-separated string into list
    # Ensure prediction column is string before splitting
    sub["predicted"] = sub["prediction"].astype(str).apply(lambda x: x.split())

    # Merge ground truth with predictions
    # We evaluate on all customers present in the validation set
    merged = valid_grouped.merge(
        sub[["customer_id", "predicted"]], on="customer_id", how="left"
    )

    # Handle missing predictions (if any) by assigning empty lists
    # This ensures the code doesn't break if a customer in valid_df is missing in sub_df
    # (though they will receive a score of 0.0)
    missing_mask = merged["predicted"].isnull()
    if missing_mask.any():
        merged.loc[missing_mask, "predicted"] = [[] for _ in range(missing_mask.sum())]

    # Calculate MAP@12
    map_score = mapk(merged["actual"].tolist(), merged["predicted"].tolist(), k=k)

    return map_score
