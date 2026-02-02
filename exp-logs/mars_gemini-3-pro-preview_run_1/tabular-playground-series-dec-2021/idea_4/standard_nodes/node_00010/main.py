"""
Self-Training Homogeneous Ensemble Strategy Implementation
"""

import os
import sys
import gc
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
from sklearn.metrics import accuracy_score

# Ensure library is in path
sys.path.append(os.getcwd())

# Import library modules
import library.config
from library.config import DATA_PATHS, PIPELINE_PARAMS, TARGET_COL, ID_COL
from library.data_utils import load_dataset, create_augmented_train
from library.model_utils import generate_submission
from library.pipeline import run_cross_validation

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seed(42)

    # --- Configuration Override for Fast Baseline ---
    # Modifying the configuration dictionary in-place to ensure propagation
    # Reducing n_estimators to ensure execution within time limits while maintaining performance
    library.config.MODEL_PARAMS["n_estimators"] = 2000

    print("Starting execution...")

    # --- 1. Load Data ---
    # Load datasets (using cache if available)
    # train_df: Training data
    # val_df: Hold-out validation data (Strictly for final evaluation)
    # test_df: Test data (for prediction and pseudo-labeling)
    train_df, val_df, test_df = load_dataset(load_cached_data=True)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Prepare Hold-out Validation Set
    val_feature_cols = [c for c in val_df.columns if c not in [TARGET_COL, ID_COL]]
    X_val_holdout = val_df[val_feature_cols]
    y_val_holdout = val_df[TARGET_COL]

    # Prepare Test Features
    test_feature_cols = [c for c in test_df.columns if c not in [TARGET_COL, ID_COL]]

    # --- 2. Stage 1: Initial Bagging (Teacher) ---
    print("\n=== Stage 1: Initial Bagging (Teacher) ===")

    # Run 5-Fold CV on original training data
    # This trains 5 models and returns them + averaged predictions on test_df
    models_s1, test_probs_s1 = run_cross_validation(
        train_df,
        test_df,
        pseudo_df=None,
        n_folds=PIPELINE_PARAMS["n_folds"],
        random_state=PIPELINE_PARAMS["random_state"],
    )

    # --- 3. Pseudo-Labeling ---
    print("\n=== Pseudo-Labeling ===")

    # Generate augmented training set (Train + High Confidence Test)
    aug_train_full = create_augmented_train(
        train_df,
        test_df,
        test_probs_s1,
        threshold=PIPELINE_PARAMS["pseudo_label_threshold"],
    )

    # Extract only the pseudo-labeled samples to pass separately to run_cross_validation
    # This ensures that we can augment the training folds while keeping validation folds clean
    # in the next CV stage (as per the "Idea" description)
    n_train = len(train_df)
    if len(aug_train_full) > n_train:
        pseudo_df = aug_train_full.iloc[n_train:].copy()
        print(f"Extracted {len(pseudo_df)} pseudo-labeled samples for Stage 2.")
    else:
        pseudo_df = None
        print("No pseudo-labels generated.")

    # Free memory
    del models_s1, aug_train_full
    gc.collect()

    # --- 4. Stage 2: Refinement (Student) ---
    print("\n=== Stage 2: Refinement (Student) ===")

    # Run 5-Fold CV on training data AUGMENTED with pseudo-labels
    # Note: pseudo_df is added to the training portion of each fold,
    # ensuring the validation portion of the fold remains pure (original train data).
    models_s2, test_probs_s2 = run_cross_validation(
        train_df,
        test_df,
        pseudo_df=pseudo_df,
        n_folds=PIPELINE_PARAMS["n_folds"],
        random_state=PIPELINE_PARAMS["random_state"],
    )

    # --- 5. Final Validation on Hold-Out Set ---
    print("\n=== Final Validation on Hold-Out Set ===")

    # Aggregate predictions from Stage 2 models
    val_probs_sum = None
    for i, model in enumerate(models_s2):
        # Predict probabilities
        probs = model.predict_proba(X_val_holdout)
        if val_probs_sum is None:
            val_probs_sum = probs
        else:
            val_probs_sum += probs

    avg_val_probs = val_probs_sum / len(models_s2)

    # Determine class labels
    if hasattr(models_s2[0], "_le"):
        classes = models_s2[0]._le.classes_
    else:
        classes = models_s2[0].classes_

    val_pred_indices = np.argmax(avg_val_probs, axis=1)
    val_preds = classes[val_pred_indices]

    # Calculate Metric
    final_metric = accuracy_score(y_val_holdout, val_preds)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # --- 6. Failure Analysis ---
    print("\n=== Failure Analysis ===")

    # Calculate Error Vector (1 if error, 0 if correct)
    errors = (val_preds != y_val_holdout).astype(int)

    # Compute Point-Biserial Correlation between Features and Error
    correlations = []
    error_vector = errors.values

    # Iterate over features to find what correlates with failure
    for col in val_feature_cols:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(X_val_holdout[col]):
            feat_values = X_val_holdout[col].values

            # Skip constant columns
            if np.std(feat_values) == 0:
                continue

            # Compute correlation
            corr = np.corrcoef(feat_values, error_vector)[0, 1]
            if not np.isnan(corr):
                correlations.append((col, corr))

    # Sort by absolute correlation (magnitude of impact)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top Features Associated with Error (Correlation):")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.4f}")

    # --- 7. Submission ---
    print("\n=== Submission Check ===")

    THRESHOLD = 0.9614708333333334

    if final_metric > THRESHOLD:
        print(f"Metric {final_metric} > {THRESHOLD}. Generating submission file...")
        # Generate submission using Stage 2 models
        generate_submission(
            models_s2, test_df[test_feature_cols], DATA_PATHS["submission_output"]
        )
    else:
        print(f"Metric {final_metric} <= {THRESHOLD}. Submission skipped.")

    print("Done.")


if __name__ == "__main__":
    main()
