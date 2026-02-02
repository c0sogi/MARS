import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from library.config import Config
from library.utils import compute_metric


def optimize_global_weights(stat_preds, deberta_preds, roberta_preds, y_true):
    """
    Optimizes the mixing weights for the three branches (Statistical, DeBERTa, RoBERTa)
    to minimize log loss on the validation set.

    Args:
        stat_preds (np.ndarray): Predictions from the statistical branch (N, 3).
        deberta_preds (np.ndarray): Predictions from the DeBERTa branch (N, 3).
        roberta_preds (np.ndarray): Predictions from the RoBERTa branch (N, 3).
        y_true (np.ndarray): Ground truth labels for the validation set.

    Returns:
        np.ndarray: Optimized weights [w_stat, w_deberta, w_roberta].
    """
    print("Optimizing Global Ensemble Weights...")

    # Organize predictions for easy access
    # Order: Statistical, DeBERTa, RoBERTa
    preds_list = [stat_preds, deberta_preds, roberta_preds]

    def objective(weights):
        # Calculate weighted average of predictions
        # blended shape: (N, 3)
        blended = np.zeros_like(preds_list[0])
        for i, w in enumerate(weights):
            blended += w * preds_list[i]

        # Calculate metric (Log Loss)
        return compute_metric(y_true, blended)

    # Initial guess: Equal weights [1/3, 1/3, 1/3]
    initial_weights = np.array([1 / 3, 1 / 3, 1 / 3])

    # Constraints: Sum of weights must equal 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # Bounds: Each weight must be between 0 and 1
    bounds = [(0, 1) for _ in range(3)]

    # Perform minimization
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"disp": False},
    )

    best_weights = result.x
    best_score = result.fun

    print(
        f"  Optimal Weights: Statistical={best_weights[0]:.4f}, "
        f"DeBERTa={best_weights[1]:.4f}, RoBERTa={best_weights[2]:.4f}"
    )
    print(f"  Best Validation Log Loss (Global Ensemble): {best_score}")

    return best_weights


def generate_submission(test_ids, stat_preds, deberta_preds, roberta_preds, weights):
    """
    Generates the final submission CSV using the optimized weights.

    Args:
        test_ids (list or np.ndarray): IDs for the test set samples.
        stat_preds (np.ndarray): Test predictions from statistical branch.
        deberta_preds (np.ndarray): Test predictions from DeBERTa branch.
        roberta_preds (np.ndarray): Test predictions from RoBERTa branch.
        weights (np.ndarray): Optimized weights [w_stat, w_deberta, w_roberta].

    Returns:
        pd.DataFrame: The submission DataFrame.
    """
    print("Generating Final Submission...")

    # Calculate weighted average
    final_preds = (
        weights[0] * stat_preds
        + weights[1] * deberta_preds
        + weights[2] * roberta_preds
    )

    # Ensure no negative probabilities due to numerical noise
    final_preds = np.maximum(final_preds, 0)

    # Construct DataFrame
    # Columns must correspond to class indices: 0->EAP, 1->HPL, 2->MWS
    submission_df = pd.DataFrame(
        {
            "id": test_ids,
            "EAP": final_preds[:, 0],
            "HPL": final_preds[:, 1],
            "MWS": final_preds[:, 2],
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return submission_df
