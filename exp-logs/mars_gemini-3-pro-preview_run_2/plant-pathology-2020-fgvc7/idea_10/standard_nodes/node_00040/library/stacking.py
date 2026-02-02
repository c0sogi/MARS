import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from library.config import CFG


def rank_normalize(preds):
    """
    Converts raw probabilities to percentile ranks (0 to 1).
    This helps mitigate calibration drift between folds and models.

    Args:
        preds (np.ndarray): Array of probabilities of shape (N,).

    Returns:
        np.ndarray: Array of percentile ranks of shape (N,).
    """
    # rankdata returns ranks from 1 to N
    # method='average' assigns the average rank to ties
    ranks = rankdata(preds, method="average")

    # Normalize to (0, 1] range
    return ranks / len(ranks)


def train_meta_learner(oof_preds_dict, targets):
    """
    Trains Logistic Regression meta-learners for Rust and Scab using ranked OOF predictions.

    Args:
        oof_preds_dict (dict): Dictionary mapping model names to OOF probability arrays.
                               Values should be np.ndarray of shape (N, 2).
                               Col 0: Rust probability, Col 1: Scab probability.
        targets (np.ndarray): Ground truth binary labels of shape (N, 2).
                              Col 0: Rust label, Col 1: Scab label.

    Returns:
        dict: A dictionary containing the trained models and metadata.
    """
    print("Training Meta-Learners (Rank-Calibrated Stacking)...")

    # Sort model names to ensure consistent feature ordering
    model_names = sorted(oof_preds_dict.keys())

    # Prepare feature matrices
    # We create separate feature sets for Rust and Scab meta-learners
    X_rust_list = []
    X_scab_list = []

    for name in model_names:
        preds = oof_preds_dict[name]

        # Rank normalize Rust predictions (Column 0)
        r_rust = rank_normalize(preds[:, 0])
        X_rust_list.append(r_rust)

        # Rank normalize Scab predictions (Column 1)
        r_scab = rank_normalize(preds[:, 1])
        X_scab_list.append(r_scab)

    # Stack features: Shape (N, num_models)
    X_rust = np.column_stack(X_rust_list)
    X_scab = np.column_stack(X_scab_list)

    # Targets
    y_rust = targets[:, 0]
    y_scab = targets[:, 1]

    # Initialize and Train Logistic Regression for Rust
    lr_rust = LogisticRegression(**CFG.meta_learner_params)
    lr_rust.fit(X_rust, y_rust)

    # Initialize and Train Logistic Regression for Scab
    lr_scab = LogisticRegression(**CFG.meta_learner_params)
    lr_scab.fit(X_scab, y_scab)

    # Bundle models
    models = {"rust": lr_rust, "scab": lr_scab, "model_names": model_names}

    # Save models using joblib
    # Ensure working directory exists
    os.makedirs(CFG.working_dir, exist_ok=True)
    save_path = os.path.join(CFG.working_dir, "meta_learner_lr.joblib")
    joblib.dump(models, save_path)
    print(f"Meta-learners saved to {save_path}")

    return models


def inference_meta_learner(models, test_preds_dict):
    """
    Generates calibrated probabilities for the test set using trained meta-learners.

    Args:
        models (dict): The dictionary returned by train_meta_learner (or loaded from disk).
        test_preds_dict (dict): Dictionary mapping model names to test probability arrays (N_test, 2).

    Returns:
        tuple: (rust_probs, scab_probs) - each is a np.ndarray of shape (N_test,).
    """
    model_names = models["model_names"]

    X_rust_list = []
    X_scab_list = []

    # Iterate in the exact same order as training
    for name in model_names:
        if name not in test_preds_dict:
            raise ValueError(
                f"Model '{name}' expected by meta-learner but not found in test predictions."
            )

        preds = test_preds_dict[name]

        # Apply Rank Normalization to Test Predictions (Transductive)
        # We rank the test set predictions relative to each other.
        r_rust = rank_normalize(preds[:, 0])
        X_rust_list.append(r_rust)

        r_scab = rank_normalize(preds[:, 1])
        X_scab_list.append(r_scab)

    X_rust = np.column_stack(X_rust_list)
    X_scab = np.column_stack(X_scab_list)

    # Predict Probabilities (Class 1)
    p_rust = models["rust"].predict_proba(X_rust)[:, 1]
    p_scab = models["scab"].predict_proba(X_scab)[:, 1]

    return p_rust, p_scab


def reconstruct_probs(rust_probs, scab_probs):
    """
    Reconstructs the 4-class probabilities from the binary Rust and Scab probabilities.

    Args:
        rust_probs (np.ndarray): Probability of Rust presence.
        scab_probs (np.ndarray): Probability of Scab presence.

    Returns:
        pd.DataFrame: DataFrame with columns ['healthy', 'multiple_diseases', 'rust', 'scab'].
    """
    # Calculate probabilities for the 4 mutually exclusive classes
    # Healthy: No Rust AND No Scab
    healthy = (1.0 - rust_probs) * (1.0 - scab_probs)

    # Rust: Rust AND No Scab
    rust_only = rust_probs * (1.0 - scab_probs)

    # Scab: No Rust AND Scab
    scab_only = (1.0 - rust_probs) * scab_probs

    # Multiple: Rust AND Scab
    multiple = rust_probs * scab_probs

    data = {
        "healthy": healthy,
        "multiple_diseases": multiple,
        "rust": rust_only,
        "scab": scab_only,
    }

    return pd.DataFrame(data)


def create_submission(test_df, rust_probs, scab_probs):
    """
    Generates the final submission file.

    Args:
        test_df (pd.DataFrame): Test dataframe containing 'image_id'.
        rust_probs (np.ndarray): Calibrated Rust probabilities.
        scab_probs (np.ndarray): Calibrated Scab probabilities.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    probs_df = reconstruct_probs(rust_probs, scab_probs)

    # Concatenate image_id with predictions
    # Ensure indices align
    submission = pd.concat(
        [test_df[["image_id"]].reset_index(drop=True), probs_df], axis=1
    )

    # Reorder columns to match sample submission format
    cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    submission = submission[cols]

    # Save to disk
    os.makedirs(CFG.submission_dir, exist_ok=True)
    submission.to_csv(CFG.submission_path, index=False)
    print(f"Submission saved to {CFG.submission_path}")

    return submission
