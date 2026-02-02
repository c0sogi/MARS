import numpy as np
import pandas as pd


def apk(actual, predicted, k=12):
    """
    Computes the average precision at k.

    Args:
        actual (set): A set of elements that are to be predicted.
        predicted (list): A list of predicted elements (order matters).
        k (int): The maximum number of predicted elements.

    Returns:
        float: The average precision at k.
    """
    if not actual:
        return 0.0

    # Slice predictions to k
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
    already_predicted = set()

    for i, p in enumerate(predicted):
        # Check if relevant and not duplicate prediction
        if p in actual and p not in already_predicted:
            already_predicted.add(p)
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    # Normalize by min(number of relevant items, k)
    return score / min(len(actual), k)


def calculate_map12(validation_df, submission_df, k=12):
    """
    Calculates MAP@12 for the given validation data and submission.

    Args:
        validation_df (pd.DataFrame): DataFrame containing true transactions.
                                      Must have 'customer_id' and 'article_id'.
        submission_df (pd.DataFrame): DataFrame containing predictions.
                                      Must have 'customer_id' and 'prediction'.
        k (int): Cutoff for MAP.

    Returns:
        float: The MAP@12 score.
    """
    print("Calculating MAP@12...")

    # 1. Prepare Ground Truth
    # Group by customer to get set of purchased items (unique items per user)
    # validation_df article_id are typically int32
    print("Aggregating ground truth...")
    ground_truth = (
        validation_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    )
    n_users = len(ground_truth)
    print(f"Ground truth contains {n_users} customers.")

    # 2. Prepare Predictions
    print("Parsing predictions...")
    # Ensure submission is indexed by customer_id for fast lookup
    submission_df = submission_df.set_index("customer_id")

    # Align predictions with ground truth customers
    # We only score customers who are in the validation set (ground_truth)
    # reindex returns NaN for missing customers, fillna('') handles them
    relevant_preds = submission_df.reindex(ground_truth.keys())
    pred_series = relevant_preds["prediction"].fillna("")

    scores = []
    cnt = 0

    print("Evaluating...")
    # Iterate over ground truth customers
    for customer_id, actual_items in ground_truth.items():
        pred_str = pred_series.at[customer_id]

        # Parse prediction string to list of integers
        # Prediction strings are like "0706016001 0706016002"
        # int conversion handles leading zeros correctly (int("0706016001") -> 706016001)
        if pred_str == "":
            predicted_items = []
        else:
            try:
                predicted_items = [int(x) for x in pred_str.split()]
            except ValueError:
                predicted_items = []

        # Calculate APK for this user
        user_score = apk(actual_items, predicted_items, k=k)
        scores.append(user_score)

        cnt += 1
        if cnt % 50000 == 0:
            print(f"Evaluated {cnt}/{n_users} users...")

    # Compute Mean Average Precision
    final_map = np.mean(scores)

    # Print full precision
    print(f"MAP@12: {final_map:.16f}")

    return final_map
