import numpy as np
import pandas as pd


def apk(actual, predicted, k=12):
    """
    Computes the Average Precision at k (AP@k) for a single user.

    This function calculates the average precision of a predicted list of items
    against a ground truth list. It treats the ground truth as a set of unique
    relevant items.

    Args:
        actual (list): A list of ground truth elements (article_ids).
        predicted (list): A list of predicted elements (article_ids).
        k (int): The maximum number of predicted elements to consider.

    Returns:
        float: The average precision at k.
    """
    # Truncate predictions to k
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    # Convert actual to a set for O(1) lookup and to handle duplicates in history
    # (Metric typically evaluates against the set of unique items bought)
    actual_set = set(actual)

    for i, p in enumerate(predicted):
        # Check if prediction is relevant and has not been predicted yet in this list
        # (Note: predicted[:i] check handles duplicate predictions if any)
        if p in actual_set and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual_set:
        return 0.0

    # The denominator is the minimum of the number of relevant items and k
    return score / min(len(actual_set), k)


def calculate_map12(valid_df, submission_df, k=12):
    """
    Calculates the Mean Average Precision @ 12 (MAP@12) for the validation set.

    This function aligns the ground truth data (from transactions) with the
    predictions (from submission format), normalizes article IDs to ensure
    matching types, and computes the mean AP@12 score.

    Args:
        valid_df (pd.DataFrame): DataFrame containing ground truth transactions.
                                 Must have 'customer_id' and 'article_id' columns.
        submission_df (pd.DataFrame): DataFrame containing predictions.
                                      Must have 'customer_id' and 'prediction' columns.
                                      'prediction' should be a space-separated string of article IDs.
        k (int): The cutoff for the metric (default 12).

    Returns:
        float: The MAP@12 score.
    """
    # Work on copies to avoid side effects
    val = valid_df.copy()
    sub = submission_df.copy()

    # Ensure customer_ids are strings for consistent merging
    val["customer_id"] = val["customer_id"].astype(str)
    sub["customer_id"] = sub["customer_id"].astype(str)

    # Aggregate ground truth: Get unique article_ids purchased by each customer
    # We use unique() (via set) because the goal is to predict the set of items bought.
    ground_truth = (
        val.groupby("customer_id")["article_id"]
        .apply(lambda x: list(set(x)))
        .reset_index()
    )
    ground_truth.columns = ["customer_id", "actual"]

    # Merge ground truth with submission
    # We use a left join on ground_truth because the metric is defined over customers
    # in the test period (represented here by valid_df).
    merged = ground_truth.merge(sub, on="customer_id", how="left")

    # Fill missing predictions with empty string (results in 0 score for that user)
    merged["prediction"] = merged["prediction"].fillna("")

    # Define a helper to compute AP for a single row
    def calc_row(row):
        # Normalize actuals to 10-digit strings to match prediction format
        # valid_df usually contains integers (e.g., 123), submission has "0000000123"
        actuals = []
        for x in row["actual"]:
            try:
                actuals.append(f"{int(x):010d}")
            except (ValueError, TypeError):
                actuals.append(str(x))

        # Parse prediction string
        preds = row["prediction"].strip().split()

        return apk(actuals, preds, k)

    # Compute AP for each customer
    merged["ap"] = merged.apply(calc_row, axis=1)

    # Return the mean
    return merged["ap"].mean()


def format_submission(customer_ids, prediction_matrix):
    """
    Formats predictions into the required submission DataFrame format.

    This function takes raw prediction arrays (integers or strings) and converts
    them into the specific space-separated, zero-padded string format required
    for the competition submission.

    Args:
        customer_ids (list or np.array): Sequence of customer IDs.
        prediction_matrix (list of lists or np.array): Sequence of prediction sequences.
                                                       Each prediction sequence contains article IDs
                                                       (int or str).

    Returns:
        pd.DataFrame: A DataFrame with columns ['customer_id', 'prediction'] ready for CSV export.
    """
    formatted_rows = []

    # Validate input lengths
    if len(customer_ids) != len(prediction_matrix):
        raise ValueError(
            f"Length mismatch: {len(customer_ids)} customers vs {len(prediction_matrix)} prediction rows."
        )

    for preds in prediction_matrix:
        # Format article IDs to 10-digit strings
        # Handles integers (123 -> "0000000123") and strings ("0123" -> "0123")
        formatted_preds = []
        for p in preds:
            try:
                # Try converting to int and back to padded string to ensure standard format
                s = f"{int(p):010d}"
            except (ValueError, TypeError):
                # Fallback for non-numeric strings (just use as is)
                s = str(p)
            formatted_preds.append(s)

        # Join with spaces
        formatted_rows.append(" ".join(formatted_preds))

    # Create DataFrame
    df = pd.DataFrame({"customer_id": customer_ids, "prediction": formatted_rows})

    return df
