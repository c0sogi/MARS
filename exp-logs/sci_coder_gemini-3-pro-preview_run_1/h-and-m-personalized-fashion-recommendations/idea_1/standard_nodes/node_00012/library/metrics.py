import pandas as pd
import numpy as np
from library.config import Config


def calculate_map12(validation_df, submission_df, k=12):
    """
    Calculates the Mean Average Precision @ 12 (MAP@12).

    This metric evaluates the order of predicted items. It is calculated as the mean
    of the Average Precision (AP) over all customers in the validation set.

    Args:
        validation_df (pd.DataFrame): DataFrame containing ground truth transactions.
            Must have 'customer_id' and 'article_id' columns.
            'article_id' is expected to be numeric (matching metadata format).
        submission_df (pd.DataFrame): DataFrame containing predictions.
            Must have 'customer_id' and 'prediction' columns.
            'prediction' should be a space-separated string of article_ids (e.g., "012345 067890").
        k (int, optional): The cutoff for precision calculation. Defaults to 12.

    Returns:
        float: The MAP@12 score.
    """
    # --------------------------------------------------------------------------
    # 1. Prepare Ground Truth
    # --------------------------------------------------------------------------
    # Group validation transactions by customer_id to get the set of purchased items.
    # We use sets for O(1) lookup during the hit check.
    ground_truth = (
        validation_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    )

    # --------------------------------------------------------------------------
    # 2. Prepare Predictions
    # --------------------------------------------------------------------------
    # Create a mapping from customer_id to prediction string for fast access.
    # We assume submission_df has unique customer_ids.
    if "customer_id" in submission_df.columns:
        # Ensure we don't modify the original dataframe
        preds_map = submission_df.set_index("customer_id")["prediction"].to_dict()
    else:
        # Fallback if customer_id is already the index
        preds_map = submission_df["prediction"].to_dict()

    # --------------------------------------------------------------------------
    # 3. Calculate MAP
    # --------------------------------------------------------------------------
    scores = []

    # We evaluate only on customers present in the validation set (ground truth).
    # Customers in submission but not in validation do not affect the score
    # (as they are not part of the summation over U).
    for customer_id, actual_items in ground_truth.items():
        m = len(actual_items)

        # If a customer in validation has no purchases (unlikely given the data source,
        # but theoretically possible if filtered), they are skipped to avoid div by zero
        # or treated as 0 depending on interpretation. Here we skip empty ground truths.
        if m == 0:
            continue

        # Retrieve prediction
        pred_str = preds_map.get(customer_id, "")

        if not isinstance(pred_str, str) or not pred_str:
            scores.append(0.0)
            continue

        # Parse predictions:
        # 1. Split string by space
        # 2. Convert to int to match the integer format of article_id in metadata/val.csv
        #    (e.g., "0108775015" -> 108775015)
        # 3. Truncate to top k
        try:
            predicted_items = [int(x) for x in pred_str.split()][:k]
        except ValueError:
            # Handle cases where prediction string might be malformed
            scores.append(0.0)
            continue

        # Calculate Average Precision (AP) for this customer
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted_items):
            # Check if predicted item is in the set of actual items
            if p in actual_items:
                num_hits += 1.0
                # Precision at rank i+1
                score += num_hits / (i + 1.0)

        # Normalize by min(m, k) per the MAP@12 formula
        ap = score / min(m, k)
        scores.append(ap)

    # Return Mean Average Precision
    if not scores:
        return 0.0

    return float(np.mean(scores))
