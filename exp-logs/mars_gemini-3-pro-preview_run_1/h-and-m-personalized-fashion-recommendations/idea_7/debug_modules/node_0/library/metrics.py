import numpy as np


def calculate_map12(ground_truth, predictions):
    """
    Computes the Mean Average Precision @ 12 (MAP@12) for recommendations.

    Args:
        ground_truth (dict): Dictionary mapping customer_id to a list of actual article_ids.
                             Values can be lists of integers or strings.
        predictions (dict): Dictionary mapping customer_id to predicted article_ids.
                            Values can be space-separated strings or lists of integers/strings.

    Returns:
        float: The calculated MAP@12 score.
    """

    # Helper function to calculate Average Precision for a single user
    def ap_at_12(actual, predicted):
        if not actual:
            return 0.0

        # Normalize predicted to a list of strings
        if isinstance(predicted, str):
            predicted = predicted.split()
        elif not isinstance(predicted, list):
            # Handle numpy arrays or other iterables
            predicted = list(predicted)

        # Truncate to top 12
        predicted = predicted[:12]

        if not predicted:
            return 0.0

        # Normalize actual items to set of 10-digit strings for O(1) lookup
        # This handles both int inputs (e.g., 123) and string inputs (e.g., "123", "000123")
        actual_set = set(str(x).zfill(10) for x in actual)

        score = 0.0
        num_hits = 0.0
        already_predicted = set()

        for i, p in enumerate(predicted):
            # Normalize prediction item
            p_str = str(p).zfill(10)

            # Skip duplicate predictions (standard MAP behavior)
            if p_str in already_predicted:
                continue
            already_predicted.add(p_str)

            if p_str in actual_set:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        # Denominator is min(len(actual), 12) per competition metric definition
        # We use len(actual) (list length) to match the standard definition where m is the number of ground truth values
        return score / min(len(actual), 12)

    # Main loop
    scores = []

    # We iterate over the ground truth users.
    # Users in predictions but not in ground truth do not contribute to the score
    # (or are effectively users with empty ground truth if we considered them, which yields 0).
    for customer_id, actual_items in ground_truth.items():
        # Get predictions for this customer, default to empty if missing
        pred_items = predictions.get(customer_id, [])

        user_score = ap_at_12(actual_items, pred_items)
        scores.append(user_score)

    if not scores:
        return 0.0

    return float(np.mean(scores))
