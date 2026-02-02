import numpy as np
import pandas as pd


def apk(actual, predicted, k=12):
    """
    Computes the Average Precision at k.

    Parameters
    ----------
    actual : list
        A list of elements that are to be predicted (ground truth).
    predicted : list
        A list of predicted elements.
    k : int, optional
        The maximum number of predicted elements.

    Returns
    -------
    score : float
        The Average Precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    # Convert actual to set for O(1) lookup and to handle uniqueness.
    # In this task, m is typically the number of unique items bought.
    actual_set = set(actual)
    m = len(actual_set)

    if m == 0:
        return 0.0

    seen_hits = set()

    for i, p in enumerate(predicted):
        if p in actual_set and p not in seen_hits:
            seen_hits.add(p)
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(m, k)


def calculate_map_at_12(val_df, sub_df):
    """
    Computes the Mean Average Precision @ 12.

    Parameters
    ----------
    val_df : pd.DataFrame
        DataFrame containing validation transactions.
        Must contain columns: 'customer_id', 'article_id'.
    sub_df : pd.DataFrame
        DataFrame containing submission predictions.
        Must contain columns: 'customer_id', 'prediction'.
        'prediction' should be a space-separated string of article_ids.

    Returns
    -------
    map_score : float
        The MAP@12 score.
    """
    # Validate columns
    if "customer_id" not in val_df.columns or "article_id" not in val_df.columns:
        raise ValueError("val_df must contain 'customer_id' and 'article_id'")
    if "customer_id" not in sub_df.columns or "prediction" not in sub_df.columns:
        raise ValueError("sub_df must contain 'customer_id' and 'prediction'")

    # 1. Prepare Ground Truth
    # Ensure article_id is int for consistent comparison
    val_df_clean = val_df.copy()
    val_df_clean["article_id"] = val_df_clean["article_id"].astype(int)

    # Group by customer_id and get list of items bought
    ground_truth = (
        val_df_clean.groupby("customer_id")["article_id"].agg(list).reset_index()
    )
    ground_truth.rename(columns={"article_id": "actual"}, inplace=True)

    # 2. Prepare Predictions
    # Merge ground_truth with submission.
    # We use a left join on ground_truth because MAP is averaged over U (users in ground truth).
    merged = ground_truth.merge(sub_df, on="customer_id", how="left")

    # Fill missing predictions with empty string (implies AP=0 for that user)
    merged["prediction"] = merged["prediction"].fillna("")

    # 3. Compute AP for each row
    actuals = merged["actual"].tolist()
    preds_raw = merged["prediction"].tolist()

    scores = []
    for actual, pred_str in zip(actuals, preds_raw):
        # Parse prediction string to list of ints
        if not pred_str:
            predicted = []
        else:
            try:
                predicted = [int(x) for x in pred_str.split()]
            except ValueError:
                predicted = []

        scores.append(apk(actual, predicted, k=12))

    return np.mean(scores)
