import numpy as np
import pandas as pd
from library.config import SUBMISSION_PATH


def apk(actual, predicted, k=12):
    """
    Computes the average precision at k.

    Parameters
    ----------
    actual : list
        A list of elements that are to be predicted (order doesn't matter)
    predicted : list
        A list of predicted elements (order matters)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
        The average precision at k over the input lists
    """
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def calculate_map12(predictions, ground_truth):
    """
    Computes the Mean Average Precision @ 12.

    Parameters
    ----------
    predictions : pd.Series or dict
        Mapping customer_id -> list of predicted article_ids
    ground_truth : pd.Series or dict
        Mapping customer_id -> list of actual article_ids

    Returns
    -------
    score : float
        The MAP@12 score
    """
    # Convert to dictionaries for faster lookup if they are Series
    if isinstance(predictions, pd.Series):
        preds_dict = predictions.to_dict()
    else:
        preds_dict = predictions

    if isinstance(ground_truth, pd.Series):
        truth_dict = ground_truth.to_dict()
    else:
        truth_dict = ground_truth

    # The metric is calculated over all customers in the ground truth
    valid_customers = list(truth_dict.keys())

    if not valid_customers:
        print("Validation MAP@12: 0.0 (No ground truth customers)")
        return 0.0

    scores = []
    for customer_id in valid_customers:
        actual = truth_dict[customer_id]
        # If no prediction for customer, assume empty list
        predicted = preds_dict.get(customer_id, [])

        # Ensure actual is a list/iterable
        if not isinstance(actual, (list, np.ndarray, tuple)):
            actual = [actual]

        # Ensure predicted is a list/iterable
        if not isinstance(predicted, (list, np.ndarray, tuple)):
            # If it's a string (e.g. "id1 id2"), split it
            if isinstance(predicted, str):
                predicted = predicted.split()
            else:
                predicted = [predicted]

        scores.append(apk(list(actual), list(predicted), k=12))

    mean_score = np.mean(scores)
    # Print full precision as requested
    print(f"Validation MAP@12: {mean_score}")
    return mean_score


def create_submission(predictions, output_path=SUBMISSION_PATH):
    """
    Formats predictions and saves to CSV in the required format.

    Parameters
    ----------
    predictions : dict or pd.Series
        Mapping customer_id -> list of predicted article_ids
    output_path : Path or str
        Path to save the submission file
    """
    if isinstance(predictions, dict):
        # Convert to list of tuples for DataFrame creation
        data = list(predictions.items())
        df = pd.DataFrame(data, columns=["customer_id", "prediction"])
    elif isinstance(predictions, pd.Series):
        df = predictions.reset_index()
        df.columns = ["customer_id", "prediction"]
    elif isinstance(predictions, pd.DataFrame):
        df = predictions.copy()
        if "customer_id" not in df.columns or "prediction" not in df.columns:
            # Fallback: assume first column is id, second is prediction
            df.columns = ["customer_id", "prediction"]
    else:
        raise ValueError("Unsupported input format for predictions")

    # Format prediction column: list to space-separated string
    def format_pred(x):
        if isinstance(x, (list, np.ndarray, tuple)):
            return " ".join(str(i) for i in x)
        return str(x)

    df["prediction"] = df["prediction"].apply(format_pred)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
