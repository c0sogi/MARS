import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Import provided library modules
import library.config as config
import library.data_utils as data_utils
from library.ensemble import StackingManager

# ==========================================
# 1. Configuration Overrides for Fast Baseline
# ==========================================
# Modify global config to ensure execution finishes quickly
config.N_FOLDS = 3  # Reduce folds from 5 to 3
config.LGBM_PARAMS["n_estimators"] = 400  # Reduce boosting rounds
config.LGBM_PARAMS["verbose"] = -1
config.NN_PARAMS["epochs"] = 5  # Reduce NN epochs
config.NN_PARAMS["batch_size"] = 4096  # Increase batch size for A100 efficiency
config.BASELINE_SCORE = 0.9604513888888889  # Threshold from task description

# Limit training data size for speed
MAX_TRAIN_SAMPLES = 100000


def perform_failure_analysis(X_val, y_val, y_pred):
    """
    Calculates and prints correlations between features and prediction errors
    on the validation set.
    """
    print("\n=== Failure Analysis on Validation Set ===")

    # Calculate binary error (0 = Correct, 1 = Incorrect)
    errors = (y_val != y_pred).astype(int)
    error_rate = errors.mean()
    print(f"Validation Error Rate: {error_rate:.6f}")

    if error_rate == 0:
        print("No errors found. Skipping correlation analysis.")
        return

    # Use a subset for correlation if validation set is huge (though 720k is manageable)
    # We use the full X_val here for accurate analysis

    # We need to ensure X_val is a DataFrame with numeric columns
    # X_val passed here is X_val_tree which has raw features + interactions

    # Calculate correlation of each feature with the Error vector
    # Using pandas corrwith or a loop is safer for memory than full correlation matrix
    print("Calculating feature correlations with Error...")

    # Select numeric features
    numeric_cols = X_val.select_dtypes(include=[np.number]).columns

    # Compute correlations efficiently
    # (X - X_mean) * (Y - Y_mean) / (std_X * std_Y)
    # But pandas .corrwith is convenient
    correlations = X_val[numeric_cols].corrwith(pd.Series(errors, index=X_val.index))

    # Sort and display
    print(
        "\nTop 5 Features positively correlated with Error (High value -> High Error):"
    )
    print(correlations.sort_values(ascending=False).head(5))

    print(
        "\nTop 5 Features negatively correlated with Error (Low value -> High Error):"
    )
    print(correlations.sort_values(ascending=True).head(5))


def main():
    # Set seeds
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    print("Initializing Stacking Pipeline...")
    manager = StackingManager()

    # ==========================================
    # 2. Data Loading & Preparation
    # ==========================================
    print("Loading preprocessed data...")
    # Load raw splits directly to handle Train/Val separation manually
    raw_data = data_utils.preprocess_data(load_cached_data=True)

    # Unpack Tree Data
    X_train_tree, y_train, X_val_tree, y_val, X_test_tree, test_ids = raw_data["tree"]
    # Unpack NN Data
    X_train_nn, _, X_val_nn, _, X_test_nn, _ = raw_data["nn"]

    print(f"Original Train Shape: {X_train_tree.shape}")
    print(f"Original Val Shape:   {X_val_tree.shape}")

    # --- Downsample Training Data ---
    if len(y_train) > MAX_TRAIN_SAMPLES:
        print(f"Downsampling Training set to {MAX_TRAIN_SAMPLES} samples...")
        indices = np.arange(len(y_train))
        _, sub_idx = train_test_split(
            indices,
            test_size=MAX_TRAIN_SAMPLES,
            stratify=y_train,
            random_state=config.SEED,
        )

        # Apply subset
        X_train_tree_sub = X_train_tree.iloc[sub_idx].reset_index(drop=True)
        X_train_nn_sub = X_train_nn.iloc[sub_idx].reset_index(drop=True)
        y_train_sub = y_train[sub_idx]
    else:
        X_train_tree_sub = X_train_tree
        X_train_nn_sub = X_train_nn
        y_train_sub = y_train

    # --- Prepare Evaluation Set (Val + Test) ---
    # We combine Val and Test so the manager predicts both in one go
    print("Combining Validation and Test sets for inference...")
    X_eval_tree = pd.concat([X_val_tree, X_test_tree], axis=0).reset_index(drop=True)
    X_eval_nn = pd.concat([X_val_nn, X_test_nn], axis=0).reset_index(drop=True)

    # Store lengths to split later
    len_val = len(X_val_tree)
    len_test = len(X_test_tree)

    # Construct Data Dictionary for Manager
    data_for_cv = {
        "tree": (X_train_tree_sub, X_eval_tree),
        "nn": (X_train_nn_sub, X_eval_nn),
        "y": y_train_sub,
    }

    # ==========================================
    # 3. Train Base Models (Level-0)
    # ==========================================
    print(f"Starting Cross-Validation on {len(y_train_sub)} samples...")
    # oof_preds: Predictions on X_train_sub (from CV)
    # eval_preds: Predictions on X_eval (Val + Test) (Averaged across folds)
    oof_lgbm, oof_nn, eval_lgbm, eval_nn = manager.cross_validate_base_models(
        data_for_cv
    )

    # ==========================================
    # 4. Meta Learner Training (Level-1)
    # ==========================================
    print("Training Meta Learner...")
    # Construct Meta Features for Training
    X_meta_train = np.hstack([oof_lgbm, oof_nn])

    # Fit Meta Learner
    manager.meta_model.fit(X_meta_train, y_train_sub)

    # ==========================================
    # 5. Validation & Analysis
    # ==========================================
    print("Performing Validation...")

    # Construct Meta Features for Evaluation
    X_meta_eval = np.hstack([eval_lgbm, eval_nn])

    # Split back into Val and Test
    X_meta_val = X_meta_eval[:len_val]
    X_meta_test = X_meta_eval[len_val:]

    # Predict on Validation Set
    val_preds = manager.meta_model.predict(X_meta_val)

    # Compute Metric
    final_metric = accuracy_score(y_val, val_preds)

    # --- REQUIRED OUTPUT ---
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    perform_failure_analysis(X_val_tree, y_val, val_preds)

    # ==========================================
    # 6. Submission
    # ==========================================
    print(f"\nChecking threshold: {final_metric} > {config.BASELINE_SCORE}")

    if final_metric > config.BASELINE_SCORE:
        print("Threshold passed. Generating submission...")

        # Predict on Test Set
        test_preds = manager.meta_model.predict(X_meta_test)

        # Generate Submission File
        manager.generate_submission(test_ids, test_preds)
        print("Submission saved.")
    else:
        print("Threshold not passed. Submission aborted.")


if __name__ == "__main__":
    main()
