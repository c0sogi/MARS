import pandas as pd
import numpy as np
import os
from library.config import Config


def apk(actual, predicted, k=12):
    """
    Computes the average precision at k.

    Args:
        actual (list): A list of elements that are to be predicted (ground truth).
        predicted (list): A list of predicted elements.
        k (int, optional): The maximum number of predicted elements. Defaults to 12.

    Returns:
        float: The average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=12):
    """
    Computes the mean average precision at k.

    Args:
        actual (list): A list of lists of elements that are to be predicted.
        predicted (list): A list of lists of predicted elements.
        k (int, optional): The maximum number of predicted elements. Defaults to 12.

    Returns:
        float: The mean average precision at k.
    """
    if not actual:
        return 0.0
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def calculate_map12(val_df, predictions, k=12):
    """
    Calculates the Mean Average Precision @ 12 (MAP@12) for the validation set.

    Args:
        val_df (pd.DataFrame): DataFrame containing 'customer_id' and 'article_id' columns representing ground truth.
        predictions (dict): Dictionary where keys are customer_ids and values are lists of predicted article_ids.
        k (int, optional): The cutoff for predictions. Defaults to 12.

    Returns:
        float: The MAP@12 score.
    """
    # Ensure we are working with list of ground truths per customer
    # Grouping by customer_id to collect all articles purchased by the customer in the validation period
    val_grouped = val_df.groupby("customer_id")["article_id"].apply(list).reset_index()

    actual_list = []
    predicted_list = []

    # Iterate over the validation customers
    # Note: We only score customers present in the validation set (ground truth exists)
    for _, row in val_grouped.iterrows():
        cust_id = row["customer_id"]
        actual_items = row["article_id"]

        # Retrieve predictions for this customer, default to empty list if missing
        pred_items = predictions.get(cust_id, [])

        actual_list.append(actual_items)
        predicted_list.append(pred_items)

    score = mapk(actual_list, predicted_list, k=k)

    print(f"Validation MAP@{k}:", score)
    return score


def format_submission(prediction_matrix, customer_ids, article_id_map):
    """
    Formats the predictions into the required CSV format and saves it.

    Args:
        prediction_matrix (np.ndarray or list): Matrix of shape (n_customers, 12) containing predicted article indices.
        customer_ids (list or np.ndarray): List of customer_id strings corresponding to the rows of prediction_matrix.
        article_id_map (dict or np.ndarray): Mapping from article index (int) to article_id (int or string).
    """
    print("Formatting submission...")

    # Ensure the submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    formatted_preds = []

    # Check if article_id_map is accessible via integer indexing (array) or key lookup (dict)
    is_dict_map = isinstance(article_id_map, dict)

    for i, row in enumerate(prediction_matrix):
        row_str_preds = []
        for val in row:
            # Map index to original article ID
            if is_dict_map:
                art_id = article_id_map[val]
            else:
                art_id = article_id_map[val]

            # Format as 10-digit string with leading zeros
            # If art_id is int: 123 -> "0000000123"
            # If art_id is str: "123" -> "0000000123"
            if isinstance(art_id, (int, np.integer)):
                row_str_preds.append(f"{art_id:010d}")
            else:
                row_str_preds.append(str(art_id).zfill(10))

        formatted_preds.append(" ".join(row_str_preds))

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {"customer_id": customer_ids, "prediction": formatted_preds}
    )

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
