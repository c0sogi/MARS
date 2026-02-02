import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from library.config import Config
from library.utils import reconstruct_4_class_probabilities


def get_ranks(predictions):
    """
    Converts raw probabilities into percentile ranks column-wise.

    Args:
        predictions (np.array): Array of shape (N_samples, N_models).

    Returns:
        np.array: Array of shape (N_samples, N_models) with values in (0, 1].
    """
    # Apply rankdata along axis 0 (samples) to get ranks (1 to N)
    # method='average' handles ties by assigning the average rank
    ranks = np.apply_along_axis(
        lambda x: rankdata(x, method="average"), axis=0, arr=predictions
    )

    # Normalize to (0, 1]
    return ranks / ranks.shape[0]


def train_meta_learner(X_train, y_train, seed=42):
    """
    Trains a Logistic Regression meta-learner.

    Args:
        X_train (np.array): Ranked OOF predictions (N_samples, N_models).
        y_train (np.array): Binary target labels (N_samples,).
        seed (int): Random seed.

    Returns:
        sklearn.linear_model.LogisticRegression: Trained model.
    """
    model = LogisticRegression(random_state=seed)
    model.fit(X_train, y_train)
    return model


def reconstruct_probabilities(rust_probs, scab_probs):
    """
    Wrapper for the reconstruction utility to map binary probabilities
    back to the 4-class format.

    Args:
        rust_probs (np.array): Probability of Rust.
        scab_probs (np.array): Probability of Scab.

    Returns:
        np.array: (N, 4) array [healthy, multiple_diseases, rust, scab].
    """
    return reconstruct_4_class_probabilities(rust_probs, scab_probs)


def run_stacking(oof_preds, test_preds_raw, y_train, load_cached_data=True):
    """
    Main pipeline for Rank-Calibrated Stacking.

    Args:
        oof_preds (np.array): OOF predictions (N_train, N_models, 2).
        test_preds_raw (np.array): Raw test predictions (N_test, N_models, 2).
        y_train (np.array): Binary ground truth targets (N_train, 2).
        load_cached_data (bool): Whether to load results from cache.

    Returns:
        np.array: Final 4-class probabilities for the test set (N_test, 4).
    """
    cache_path = os.path.join(Config.WORKING_DIR, "stacked_test_probs.npy")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached stacked predictions from {cache_path}")
            return np.load(cache_path)
        except Exception:
            print("Failed to load cache, re-computing...")

    print("Running Rank-Calibrated Stacking...")

    num_test = test_preds_raw.shape[0]
    final_binary_preds = np.zeros((num_test, 2))  # [rust, scab]
    target_names = Config.TARGET_COLS  # ["rust", "scab"]

    # 2. Process each target independently
    for i, target_name in enumerate(target_names):
        print(f"Processing Stacking for target: {target_name}")

        # Extract features (predictions from all base models for this target)
        X_train_feat = oof_preds[:, :, i]  # (N_train, N_models)
        X_test_feat = test_preds_raw[:, :, i]  # (N_test, N_models)
        y_train_target = y_train[:, i]

        # Rank Normalization
        # This mitigates calibration drift between folds
        X_train_ranked = get_ranks(X_train_feat)
        X_test_ranked = get_ranks(X_test_feat)

        # Train Meta-Learner
        meta_model = train_meta_learner(
            X_train_ranked, y_train_target, seed=Config.SEED
        )

        # Print coefficients (Full precision as requested)
        print(f"  Intercept: {meta_model.intercept_[0]}")
        print(f"  Coefficients: {meta_model.coef_[0]}")

        # Predict on Test Set
        # predict_proba returns [prob_0, prob_1], we want prob_1
        final_binary_preds[:, i] = meta_model.predict_proba(X_test_ranked)[:, 1]

    # 3. Reconstruct 4-Class Probabilities
    # Map independent Rust/Scab probabilities to Healthy/Multiple/Rust/Scab
    final_probs_4class = reconstruct_probabilities(
        final_binary_preds[:, 0], final_binary_preds[:, 1]
    )

    # 4. Cache Results
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, final_probs_4class)

    return final_probs_4class


def generate_submission(final_probs, output_path=Config.SUBMISSION_PATH):
    """
    Generates the submission file from the final probabilities.

    Args:
        final_probs (np.array): (N_test, 4) probabilities.
        output_path (str): Path to save the CSV.
    """
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    submission_df = pd.DataFrame(
        {
            "image_id": test_df["image_id"],
            "healthy": final_probs[:, 0],
            "multiple_diseases": final_probs[:, 1],
            "rust": final_probs[:, 2],
            "scab": final_probs[:, 3],
        }
    )

    # Ensure correct column order
    cols_order = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    submission_df = submission_df[cols_order]

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
