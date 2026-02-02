import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
import os
from typing import Dict, Tuple, List, Union

from library.config import Config


def calculate_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def _find_best_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, num_steps: int = 200
) -> Tuple[float, float]:
    """
    Finds the probability threshold that maximizes MCC using a linear search.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        num_steps: Number of steps for linear search (default 200 -> 0.005 step).

    Returns:
        Tuple[float, float]: (best_threshold, best_mcc)
    """
    best_mcc = -1.0
    best_thresh = 0.5

    # Generate thresholds from 0 to 1
    thresholds = np.linspace(0, 1, num_steps + 1)

    # Iterate through thresholds
    for thresh in thresholds:
        # Apply threshold
        y_pred = (y_prob >= thresh).astype(int)

        # Calculate MCC
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    return best_thresh, best_mcc


def optimize_thresholds(
    y_true_A: np.ndarray,
    y_prob_A: np.ndarray,
    y_true_B: np.ndarray,
    y_prob_B: np.ndarray,
) -> Dict[str, float]:
    """
    Optimizes thresholds independently for Stream A (Player-Player) and Stream B (Player-Ground).

    Args:
        y_true_A: Validation labels for Stream A.
        y_prob_A: Validation probabilities for Stream A.
        y_true_B: Validation labels for Stream B.
        y_prob_B: Validation probabilities for Stream B.

    Returns:
        Dict: Dictionary containing best thresholds and MCC scores.
    """
    print("Optimizing thresholds for Stream A (Interaction)...")
    thresh_A, mcc_A = _find_best_threshold(y_true_A, y_prob_A, num_steps=200)
    print(f"Stream A - Best Threshold: {thresh_A}")
    print(f"Stream A - Best MCC: {mcc_A}")

    print("Optimizing thresholds for Stream B (Impact)...")
    thresh_B, mcc_B = _find_best_threshold(y_true_B, y_prob_B, num_steps=200)
    print(f"Stream B - Best Threshold: {thresh_B}")
    print(f"Stream B - Best MCC: {mcc_B}")

    # Calculate Global MCC on Validation Set for reference
    y_true_all = np.concatenate([y_true_A, y_true_B])

    y_pred_A = (y_prob_A >= thresh_A).astype(int)
    y_pred_B = (y_prob_B >= thresh_B).astype(int)
    y_pred_all = np.concatenate([y_pred_A, y_pred_B])

    global_mcc = matthews_corrcoef(y_true_all, y_pred_all)
    print(f"Global Combined Validation MCC: {global_mcc}")

    return {
        "thresh_A": thresh_A,
        "mcc_A": mcc_A,
        "thresh_B": thresh_B,
        "mcc_B": mcc_B,
        "global_mcc": global_mcc,
    }


def generate_submission(
    ids_A: np.ndarray,
    probs_A: np.ndarray,
    ids_B: np.ndarray,
    probs_B: np.ndarray,
    thresh_A: float,
    thresh_B: float,
    output_path: str = Config.SUBMISSION_FILE_PATH,
):
    """
    Applies thresholds to test probabilities, merges with sample submission structure,
    and generates the final CSV file.

    Args:
        ids_A: Contact IDs for Stream A.
        probs_A: Probabilities for Stream A.
        ids_B: Contact IDs for Stream B.
        probs_B: Probabilities for Stream B.
        thresh_A: Optimized threshold for Stream A.
        thresh_B: Optimized threshold for Stream B.
        output_path: Path to save the submission CSV.
    """
    print(f"Generating submission file at {output_path}...")

    # Stream A Predictions
    preds_A = (probs_A >= thresh_A).astype(int)
    df_A = pd.DataFrame({"contact_id": ids_A, "contact": preds_A})

    # Stream B Predictions
    preds_B = (probs_B >= thresh_B).astype(int)
    df_B = pd.DataFrame({"contact_id": ids_B, "contact": preds_B})

    # Combine predictions
    df_sub = pd.concat([df_A, df_B], axis=0)

    # Ensure all contact_ids from sample_submission are present and in correct order
    if os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Merge with sample submission to ensure order and completeness
        # We use left join on sample_sub to keep its order
        final_sub = sample_sub[["contact_id"]].merge(
            df_sub, on="contact_id", how="left"
        )

        # Fill missing values with 0 (no contact) as a fallback
        # This handles any IDs that might have been filtered out during preprocessing
        missing_count = final_sub["contact"].isnull().sum()
        if missing_count > 0:
            print(
                f"Warning: {missing_count} contact_ids were missing predictions. Filling with 0."
            )
            final_sub["contact"] = final_sub["contact"].fillna(0).astype(int)
        else:
            final_sub["contact"] = final_sub["contact"].astype(int)

    else:
        # If sample submission not found, just save what we have
        print("Warning: Sample submission not found. Saving predictions as is.")
        final_sub = df_sub

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_sub.to_csv(output_path, index=False)
    print("Submission saved successfully.")
