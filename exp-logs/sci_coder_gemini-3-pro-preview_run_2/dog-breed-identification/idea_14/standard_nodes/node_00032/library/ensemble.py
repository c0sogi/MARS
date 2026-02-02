import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
import library.config as config


def optimize_ensemble_weights(
    preds_a: np.ndarray, preds_b: np.ndarray, targets: np.ndarray
):
    """
    Finds the optimal scalar weights (w_a, w_b) that minimize Log Loss
    for the ensemble prediction: P_final = w_a * P_a + w_b * P_b.
    Constraints: w_a + w_b = 1, 0 <= w_a, w_b <= 1.

    Args:
        preds_a (np.ndarray): Probability matrix from Stream A (N_samples, N_classes).
        preds_b (np.ndarray): Probability matrix from Stream B (N_samples, N_classes).
        targets (np.ndarray): True class indices (N_samples,).

    Returns:
        tuple: (w_a, w_b) - The optimal weights.
    """
    print("Optimizing ensemble weights...")

    # Clip predictions to avoid log(0) issues, though log_loss handles this internally usually.
    # Keeping it raw for the optimizer to handle via log_loss function.

    def objective(w):
        # w is the weight for Stream A
        # (1 - w) is the weight for Stream B
        weighted_preds = w * preds_a + (1 - w) * preds_b
        # Normalize just in case, though convex combination of valid probas is valid.
        # log_loss requires inputs to be valid probabilities.
        return log_loss(targets, weighted_preds)

    # Optimize w within [0, 1]
    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")

    best_w_a = result.x
    best_w_b = 1.0 - best_w_a

    print(f"Optimization Complete.")
    print(f"Optimal Weights -> Stream A: {best_w_a:.4f}, Stream B: {best_w_b:.4f}")
    print(f"Optimized Validation Log Loss: {result.fun}")

    return best_w_a, best_w_b


def compute_weighted_prediction(
    preds_a: np.ndarray, preds_b: np.ndarray, w_a: float, w_b: float
):
    """
    Computes the weighted average of predictions from two streams.

    Args:
        preds_a (np.ndarray): Probability matrix from Stream A.
        preds_b (np.ndarray): Probability matrix from Stream B.
        w_a (float): Weight for Stream A.
        w_b (float): Weight for Stream B.

    Returns:
        np.ndarray: The combined probability matrix.
    """
    return w_a * preds_a + w_b * preds_b


def generate_submission(
    test_ids: np.ndarray,
    predictions: np.ndarray,
    output_path: str = config.SUBMISSION_PATH,
):
    """
    Generates the submission CSV file.

    Args:
        test_ids (np.ndarray): Array of test image IDs.
        predictions (np.ndarray): Matrix of predicted probabilities (N_test, N_classes).
        output_path (str): Path to save the CSV file.
    """
    print(f"Generating submission file at {output_path}...")

    # 1. Get Class Names (Breeds)
    # The model was trained with classes sorted alphabetically (see library.dataset.get_class_mapping).
    # We must retrieve the breeds in the same order to create the header.
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Train metadata not found at {config.TRAIN_METADATA_PATH}"
        )

    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    unique_breeds = sorted(df_train["breed"].unique().tolist())

    if len(unique_breeds) != predictions.shape[1]:
        raise ValueError(
            f"Mismatch between number of breeds ({len(unique_breeds)}) "
            f"and prediction columns ({predictions.shape[1]})."
        )

    # 2. Create DataFrame
    # Column 1 is 'id', subsequent columns are breed names
    submission_df = pd.DataFrame(predictions, columns=unique_breeds)
    submission_df.insert(0, "id", test_ids)

    # 3. Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print("Submission file saved successfully.")
