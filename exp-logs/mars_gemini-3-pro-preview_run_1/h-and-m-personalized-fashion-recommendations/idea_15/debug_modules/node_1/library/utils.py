import numpy as np
import pandas as pd
from library import config


def calculate_decay_weights(t_dat_series, reference_date, decay_rate):
    """
    Calculates temporal weights based on the number of days elapsed between
    the transaction date and the reference date using a power-law decay.

    Formula: weight = (1 + days_elapsed) ** (-decay_rate)

    Args:
        t_dat_series (pd.Series): Series containing transaction dates. Can be strings or datetime objects.
        reference_date (pd.Timestamp): The reference date to calculate elapsed days from.
        decay_rate (float): The exponent for the decay function.

    Returns:
        np.ndarray: Array of weights with type defined in config.FLOAT_DTYPE.
    """
    # Convert to datetime if necessary
    if not np.issubdtype(t_dat_series.dtype, np.datetime64):
        t_dat_series = pd.to_datetime(t_dat_series)

    # Calculate days elapsed
    # reference_date is typically the day AFTER the training period ends.
    delta = reference_date - t_dat_series

    # Handle both Series (needs .dt accessor) and Index (has direct .days attribute)
    if hasattr(delta, "dt"):
        days_elapsed = delta.dt.days
    else:
        days_elapsed = delta.days

    # Ensure no negative values (though data should be historical)
    days_elapsed = days_elapsed.clip(lower=0)

    # Apply power law decay
    # Adding 1.0 to avoid division by zero or log(0) issues,
    # and to ensure the most recent day has weight 1.0 (if rate > 0).
    weights = np.power(1.0 + days_elapsed, -decay_rate)

    return weights.values.astype(config.FLOAT_DTYPE)


def apk(actual, predicted, k=12):
    """
    Computes the Average Precision at k (AP@k) for a single user.

    Args:
        actual (list): List of ground truth items (article_ids).
        predicted (list): List of predicted items (article_ids).
        k (int): Maximum number of predictions to evaluate.

    Returns:
        float: The AP@k score.
    """
    if not actual:
        return 0.0

    # Truncate predictions to k
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    # Use a set for O(1) lookup of relevance
    actual_set = set(actual)

    for i, p in enumerate(predicted):
        if p in actual_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    # The metric is defined as sum(P(k) * rel(k)) / min(m, 12)
    # where m is the number of ground truth values.
    return score / min(len(actual), k)


def calculate_map12(valid_df, submission_df):
    """
    Calculates the Mean Average Precision @ 12 (MAP@12) for the validation set.

    Args:
        valid_df (pd.DataFrame): DataFrame containing validation transactions.
                                 Must contain 'customer_id' and 'article_id'.
        submission_df (pd.DataFrame): DataFrame containing predictions.
                                      Must contain 'customer_id' and 'prediction'.
                                      'prediction' should be a space-separated string of article IDs.

    Returns:
        float: The MAP@12 score.
    """
    # Ensure we are working with the correct types
    # valid_df 'article_id' are typically int32 based on metadata loading
    # submission_df 'prediction' are strings "id1 id2 ..."

    # 1. Prepare Ground Truth
    # Group by customer_id to get list of purchased articles
    ground_truth = valid_df.groupby("customer_id")["article_id"].apply(list).to_dict()

    # 2. Prepare Predictions
    # We create a dictionary for fast lookup
    if "customer_id" in submission_df.columns:
        predictions_map = submission_df.set_index("customer_id")["prediction"].to_dict()
    else:
        # Assuming index is customer_id if column not found
        predictions_map = submission_df["prediction"].to_dict()

    # 3. Calculate AP@12 for each customer in the validation set
    scores = []

    for customer_id, actual_items in ground_truth.items():
        # Get prediction string, default to empty
        pred_str = predictions_map.get(customer_id, "")

        # Parse prediction string to list of integers
        if pd.isna(pred_str) or pred_str == "":
            predicted_items = []
        else:
            try:
                predicted_items = [int(x) for x in pred_str.split()]
            except ValueError:
                # Handle cases where conversion fails or strings are malformed
                predicted_items = []

        # Compute AP
        user_score = apk(actual_items, predicted_items, k=12)
        scores.append(user_score)

    # 4. Compute Mean
    if not scores:
        return 0.0

    return np.mean(scores)
