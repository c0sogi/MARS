import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

# Import from provided libraries
from library.utils import set_seed, save_submission
from library.trainer import Trainer
from library.feature_engineering import build_jbpce_pipeline


def main():
    # 1. Setup
    set_seed(42)

    # 2. Data Preparation
    print("Initializing Trainer and preparing data...")
    trainer = Trainer()
    # load_cached_data=True allows using pre-computed embeddings if available in ./working
    X_full, y_full, X_test, test_ids, dims = trainer.prepare_data(load_cached_data=True)

    # 3. Stratified Cross-Validation Loop
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Arrays to store Out-of-Fold predictions and Test predictions
    oof_preds = np.zeros(len(y_full))
    test_preds_sum = np.zeros(len(X_test))

    # Define Parameter Grid (Consistent with library/trainer.py)
    param_grid = {
        "preprocessor__emb_joint_pca__n_components": [100, 150, 200],
        "clf__estimator__C": [0.1, 1.0, 10.0],
        "clf__estimator__class_weight": ["balanced", None],
        "clf__n_estimators": [20],
        "clf__max_samples": [1.0],
        "clf__bootstrap": [True],
    }

    print(f"Starting {n_splits}-Fold Stratified CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]

        # Build Pipeline using the provided factory function
        pipeline = build_jbpce_pipeline(
            emb_dim_a=dims["emb_a"], emb_dim_b=dims["emb_b"], random_state=42
        )

        # Grid Search for Hyperparameter Tuning
        # Using n_jobs=10 to utilize available vCPUs
        grid = GridSearchCV(
            pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=10, verbose=0
        )

        grid.fit(X_train_fold, y_train_fold)

        best_model = grid.best_estimator_
        print(f"  Best Internal CV AUC: {grid.best_score_:.4f}")
        print(f"  Best Params: {grid.best_params_}")

        # Validation Inference (OOF)
        val_probs = best_model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = val_probs

        fold_auc = roc_auc_score(y_val_fold, val_probs)
        print(f"  Fold Validation AUC: {fold_auc:.4f}")

        # Test Inference
        test_probs = best_model.predict_proba(X_test)[:, 1]
        test_preds_sum += test_probs

    # 4. Final Evaluation
    final_auc = roc_auc_score(y_full, oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude (absolute difference between truth and prediction)
    errors = np.abs(y_full - oof_preds)

    # Extract Metadata for Correlation Analysis
    # Metadata columns are the last N columns in X_full
    meta_dim = dims["meta"]
    X_meta = X_full[:, -meta_dim:]

    # Metadata feature names from library/utils.py
    meta_col_names = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "unix_timestamp_of_request",
    ]

    print("Correlation between Error Magnitude and Metadata Features:")
    for i, col_name in enumerate(meta_col_names):
        feat_vals = X_meta[:, i]
        # Handle constant columns to avoid NaN correlation
        if np.std(feat_vals) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(errors, feat_vals)[0, 1]
        print(f"{col_name:<60}: {corr:.4f}")

    # 6. Submission
    target_threshold = 0.7190361601447052
    if final_auc > target_threshold:
        print(
            f"\nValidation metric ({final_auc}) meets threshold ({target_threshold}). Generating submission..."
        )
        avg_test_preds = test_preds_sum / n_splits
        save_submission(
            test_ids, avg_test_preds, output_path="./submission/submission.csv"
        )
    else:
        print(
            f"\nValidation metric ({final_auc}) does NOT meet threshold ({target_threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
