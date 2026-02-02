import numpy as np
import pandas as pd
from scipy.optimize import minimize
from library.utils import compute_qwk


def apply_thresholds(y_pred, thresholds):
    """
    Applies learned thresholds to continuous predictions to obtain integer classes (1-6).

    Args:
        y_pred (np.array or list): Continuous model predictions.
        thresholds (np.array or list): A list of 5 thresholds used to separate the 6 classes.

    Returns:
        np.array: Integer predictions in the range [1, 6].
    """
    y_pred = np.array(y_pred)
    # Ensure thresholds are sorted to maintain monotonic logic
    thresholds = np.sort(thresholds)

    # Initialize all predictions to the lowest class (1)
    y_pred_int = np.ones_like(y_pred, dtype=int)

    # Apply thresholds sequentially
    # Logic:
    # If pred < t0, class is 1 (already set)
    # If pred >= t0, class becomes 2
    # If pred >= t1, class becomes 3
    # ...
    # If pred >= t4, class becomes 6
    for i, t in enumerate(thresholds):
        y_pred_int[y_pred >= t] = i + 2

    return y_pred_int


def optimize_thresholds(y_true, y_pred, initial_thresholds=None):
    """
    Optimizes decision thresholds using the Nelder-Mead algorithm to maximize
    Quadratic Weighted Kappa (QWK) on the provided ground truth and predictions.

    Args:
        y_true (np.array): Ground truth labels (integers 1-6).
        y_pred (np.array): Continuous predictions from the model/meta-learner.
        initial_thresholds (list, optional): Initial guess for thresholds.
                                             Defaults to standard rounding boundaries.

    Returns:
        np.array: The optimized thresholds.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Default initial thresholds: [1.5, 2.5, 3.5, 4.5, 5.5]
    # These correspond to standard rounding for classes 1 through 6
    if initial_thresholds is None:
        initial_thresholds = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    def objective_function(thresholds):
        """
        Objective function to minimize. Returns negative QWK.
        """
        # Nelder-Mead does not guarantee order, so we sort thresholds explicitly
        sorted_thresholds = np.sort(thresholds)

        # Discretize predictions
        preds_int = apply_thresholds(y_pred, sorted_thresholds)

        # Calculate QWK
        score = compute_qwk(y_true, preds_int)

        # Return negative score because scipy minimizes the objective
        return -score

    print("Starting threshold optimization using Nelder-Mead...")

    # Run optimization
    result = minimize(
        objective_function,
        initial_thresholds,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-4},
    )

    best_thresholds = np.sort(result.x)
    best_score = -result.fun

    print("Optimization complete.")
    print(f"Best Validation QWK: {best_score}")
    print(f"Optimized Thresholds: {best_thresholds}")

    return best_thresholds


def generate_submission(essay_ids, predictions, output_path):
    """
    Generates and saves the submission file in the required format.

    Args:
        essay_ids (list or np.array): The IDs of the essays.
        predictions (list or np.array): The predicted integer scores.
        output_path (str): The file path to save the CSV.
    """
    # Create DataFrame
    submission_df = pd.DataFrame({"essay_id": essay_ids, "score": predictions})

    # Ensure scores are integers
    submission_df["score"] = submission_df["score"].astype(int)

    # Save to CSV without index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
    print(f"Submission shape: {submission_df.shape}")
    print("First 5 rows:")
    print(submission_df.head())
