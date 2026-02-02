import pandas as pd
import numpy as np


def calculate_map12(target_df, submission_df):
    """
    Calculates the Mean Average Precision @ 12 (MAP@12) for the recommendation task.

    The metric is defined as:
    MAP@12 = (1/U) * sum_{u=1}^U ( (1/min(m, 12)) * sum_{k=1}^min(n, 12) (P(k) * rel(k)) )

    Where:
    - U is the number of customers in the evaluation set.
    - m is the number of ground truth items for a customer.
    - n is the number of predictions for a customer.
    - P(k) is the precision at cutoff k.
    - rel(k) is an indicator function equaling 1 if the item at rank k is relevant.

    Args:
        target_df (pd.DataFrame): Ground truth DataFrame. Must contain:
            - 'customer_id': Unique identifier for the customer.
            - 'article_id': Identifier for the purchased item (int or string).
        submission_df (pd.DataFrame): Prediction DataFrame. Must contain:
            - 'customer_id': Unique identifier for the customer.
            - 'prediction': Space-separated string of predicted article IDs.

    Returns:
        float: The MAP@12 score.
    """
    # Create copies of the relevant columns to avoid mutating the original DataFrames
    truth = target_df[["customer_id", "article_id"]].copy()
    preds = submission_df[["customer_id", "prediction"]].copy()

    # --- 1. Format Ground Truth ---
    # The submission format requires 10-digit strings (e.g., "0123456789").
    # The input data often has article_id as int (e.g., 123456789).
    # We must standardize the ground truth to match the predictions.
    if pd.api.types.is_numeric_dtype(truth["article_id"]):
        truth["article_id"] = truth["article_id"].apply(lambda x: f"{x:010d}")
    else:
        truth["article_id"] = truth["article_id"].astype(str).str.zfill(10)

    # Group ground truth by customer to get the set of relevant items.
    # Using a set handles duplicates in the ground truth (if a user bought the same item twice,
    # it is usually treated as a single relevant item for retrieval metrics).
    truth_grouped = truth.groupby("customer_id")["article_id"].apply(set).reset_index()
    truth_grouped.rename(columns={"article_id": "actual"}, inplace=True)

    # --- 2. Format Predictions ---
    # Convert the space-separated string of predictions into a list of strings.
    # Example: "id1 id2 id3" -> ["id1", "id2", "id3"]
    preds["prediction"] = preds["prediction"].astype(str).str.split()

    # --- 3. Merge ---
    # We evaluate over all customers present in the ground truth (target_df).
    # A Left Join ensures we have a row for every customer we need to score.
    merged = truth_grouped.merge(preds, on="customer_id", how="left")

    # --- 4. Compute AP@12 ---
    def compute_user_ap(row):
        actual = row["actual"]
        predicted = row["prediction"]

        # If there are no ground truth items, AP is technically undefined or 0.
        # In this dataset, target_df should only contain valid transactions.
        if not actual:
            return 0.0

        # If the user exists in ground truth but has no predictions (NaN or empty list), score is 0.
        if not isinstance(predicted, list):
            return 0.0

        # We only evaluate the top 12 predictions
        predicted = predicted[:12]

        score = 0.0
        num_hits = 0.0

        for k, item in enumerate(predicted):
            # Check relevance
            if item in actual:
                num_hits += 1.0
                # Precision @ k = (number of relevant items in top k) / k
                # Note: k is 0-indexed here, so we divide by (k + 1)
                score += num_hits / (k + 1.0)

        # Average Precision = Sum(Precision@k * rel(k)) / min(m, 12)
        # m = len(actual)
        return score / min(len(actual), 12)

    # Apply the computation row-wise
    ap_scores = merged.apply(compute_user_ap, axis=1)

    # --- 5. Compute Mean ---
    map_score = ap_scores.mean()

    return map_score
