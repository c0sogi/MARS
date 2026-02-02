import pandas as pd
import numpy as np
from library.utils import Timer


def apk(actual, predicted, k=12):
    """
    Computes the average precision at k.

    Args:
        actual (list): A list of ground truth elements (article_ids).
                       Duplicate items are preserved for the normalization factor min(m, 12).
        predicted (list): A list of predicted elements (article_ids).
        k (int): The maximum number of predictions to consider.

    Returns:
        float: The average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    # Use a set for O(1) relevance lookups
    actual_set = set(actual)

    for i, p in enumerate(predicted):
        # Check if the predicted item is relevant and hasn't been predicted at a higher rank
        if p in actual_set and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    # The metric is normalized by the minimum of the number of ground truth items and k
    return score / min(len(actual), k)


def mapk(actual, predicted, k=12):
    """
    Computes the mean average precision at k.

    Args:
        actual (list of lists): The ground truth lists.
        predicted (list of lists): The predicted lists.
        k (int): The maximum number of predictions to consider.

    Returns:
        float: The mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


class Evaluator:
    """
    Evaluates the performance of the recommender system using the MAP@12 metric.
    """

    def __init__(self):
        pass

    def calculate_map12(self, val_df, submission_df):
        """
        Calculates MAP@12 for the validation set.

        Args:
            val_df (pd.DataFrame): Ground truth transactions (Long format).
                                   Must contain 'customer_id' and 'article_id'.
            submission_df (pd.DataFrame): Predictions (Submission format).
                                          Must contain 'customer_id' and 'prediction' (space-separated string).

        Returns:
            float: The MAP@12 score.
        """
        with Timer("MAP@12 Calculation"):
            # 1. Prepare Ground Truth
            # Ensure IDs are strings to match submission format
            val_df = val_df.copy()
            if val_df["article_id"].dtype != object:
                val_df["article_id"] = val_df["article_id"].astype(str)

            # Group by customer to get the list of items purchased
            # Note: We preserve duplicates in the list because 'm' in min(m, 12) counts them
            ground_truth = (
                val_df.groupby("customer_id")["article_id"].apply(list).reset_index()
            )
            ground_truth.rename(columns={"article_id": "actual"}, inplace=True)

            # 2. Prepare Predictions
            submission_df = submission_df.copy()
            submission_df["prediction"] = submission_df["prediction"].astype(str)

            # 3. Merge Ground Truth with Predictions
            # We left join on ground_truth because the metric is defined over customers
            # who made purchases in the test period (val_df).
            merged = ground_truth.merge(submission_df, on="customer_id", how="left")

            # Fill missing predictions with empty string (implies empty list later)
            merged["prediction"] = merged["prediction"].fillna("")

            # 4. Compute Metric
            actual_list = merged["actual"].tolist()
            # Split prediction strings into lists
            pred_list = merged["prediction"].str.split().tolist()

            # Handle case where split returns None or empty lists incorrectly
            # (str.split() on empty string returns [], which is correct)

            score = mapk(actual_list, pred_list, k=12)

            # Print full precision as requested
            print(f"MAP@12 Score: {score}")

            return score
