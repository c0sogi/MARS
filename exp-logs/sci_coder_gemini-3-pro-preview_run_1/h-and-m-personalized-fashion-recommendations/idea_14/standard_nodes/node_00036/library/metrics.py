import pandas as pd
import numpy as np
import os
from library.config import Config


def get_validation_truth(load_cached_data=True):
    """
    Loads and aggregates the validation data to create the ground truth labels.
    Returns a pandas Series where index is customer_id and value is list of article_ids.

    Implements caching to ./working/idea_14/validation_truth.parquet to speed up
    repeated evaluations.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "validation_truth.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Convert space-separated string back to list of strings
            return df.set_index("customer_id")["article_ids"].apply(
                lambda x: x.split() if x else []
            )
        except Exception:
            # If cache is corrupt or unreadable, fall back to re-computing
            pass

    # 2. Compute from scratch
    if not os.path.exists(Config.VAL_CSV):
        raise FileNotFoundError(f"Validation data not found at {Config.VAL_CSV}")

    # Load validation data
    # Ensure article_id is string to match prediction format (e.g. "0706016001")
    df_val = pd.read_csv(Config.VAL_CSV, dtype={"article_id": str})

    # Aggregate purchases by customer into lists
    # Note: We keep duplicates in the list to correctly calculate 'm' (number of purchases)
    # as per the metric definition, although standard AP checks set membership for relevance.
    truth = df_val.groupby("customer_id")["article_id"].apply(list)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Convert list to space-separated string for Parquet storage
    # Parquet handles strings much better than object columns of lists
    save_df = truth.to_frame(name="article_ids")
    save_df["article_ids"] = save_df["article_ids"].apply(lambda x: " ".join(x))
    save_df.reset_index().to_parquet(cache_path, index=False)

    return truth


def apk(actual, predicted, k=12):
    """
    Computes the Average Precision at k (AP@k) for a single user.

    Args:
        actual : list
            A list of ground truth article_ids (order doesn't matter for set check,
            but length matters for normalization).
        predicted : list
            A list of predicted article_ids (order matters).
        k : int
            The maximum number of predicted elements to consider.

    Returns:
        score : float
            The average precision at k.
    """
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    # Use a set for O(1) relevance lookups
    actual_set = set(actual)

    # Track items already credited to avoid double-counting precision for duplicate predictions
    predicted_set = set()

    for i, p in enumerate(predicted):
        # We only reward the first time a correct item is predicted
        if p in actual_set and p not in predicted_set:
            predicted_set.add(p)
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    # The metric definition normalizes by min(m, 12) where m is the number of ground truth values.
    # len(actual) gives m.
    return score / min(len(actual), k)


def mapk(actual, predicted, k=12):
    """
    Computes the Mean Average Precision at k (MAP@k).

    Args:
        actual : list of lists
            Ground truth lists for all users.
        predicted : list of lists
            Predicted lists for all users.
        k : int
            Cutoff parameter.

    Returns:
        score : float
            The mean of AP@k scores.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def calculate_map12(predictions_df, load_cached_data=True):
    """
    Calculates the MAP@12 score for the provided predictions against the validation set.

    Args:
        predictions_df: pd.DataFrame
            DataFrame containing columns ['customer_id', 'prediction'].
            'prediction' should be a space-separated string of article_ids.
        load_cached_data: bool
            Whether to use cached validation ground truth.

    Returns:
        float: The MAP@12 score.
    """
    # 1. Load Ground Truth
    truth_series = get_validation_truth(load_cached_data=load_cached_data)

    # 2. Align Predictions with Validation Set
    # We evaluate on all customers present in the validation set.
    valid_customers = truth_series.index

    # Set index for efficient joining
    preds = predictions_df.set_index("customer_id")

    # Reindex predictions to match validation customers exactly.
    # Customers in validation set but missing from predictions get NaN -> fill with empty string.
    # Customers in predictions but not in validation set are ignored (standard hold-out logic).
    preds = preds.reindex(valid_customers).fillna("")

    # 3. Prepare Lists for Metric Calculation
    # Convert space-separated prediction strings to lists
    predicted_lists = preds["prediction"].astype(str).str.split().tolist()

    # Get actual lists
    actual_lists = truth_series.tolist()

    # 4. Compute Metric
    score = mapk(actual_lists, predicted_lists, k=12)

    print(f"Validation MAP@12: {score}")

    return score
