import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.lexical_branch import run_tfidf_ridge
from library.semantic_trainer import run_semantic_training
from library.postprocessing import (
    optimize_thresholds,
    apply_thresholds,
    generate_submission,
)

# --- Configuration for Fast Baseline Execution ---
# We override Config parameters to ensure the code runs within the time limit
# and serves as a quick verification of the pipeline.
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 2000  # Subset size for fast execution
Config.NUM_EPOCHS = 2  # Reduced epochs for speed
Config.N_FOLDS = 5  # Maintain 5 folds for structural integrity
Config.TRAIN_BATCH_SIZE = 2  # Fits on A100 with Large model
Config.GRADIENT_ACCUMULATION_STEPS = 8

# Ensure reproducibility
seed_everything(Config.SEED)


def main():
    print("=========================================")
    print("   Hybrid Stacking Network Orchestrator  ")
    print("=========================================")

    # --- Step 1: Lexical Branch Execution ---
    print("\n[1/6] Running Lexical Branch (TF-IDF + Ridge)...")
    # We disable cache loading to ensure we use the specific debug subset defined above
    df_train_lex, df_val_lex, df_test_lex = run_tfidf_ridge(
        load_cached_data=False, debug=True
    )

    # --- Step 2: Semantic Branch Execution ---
    print("\n[2/6] Running Semantic Branch (DeBERTa-v3-Large)...")
    df_train_sem, df_val_sem, df_test_sem = run_semantic_training(
        debug=True, load_cached_data=False
    )

    # --- Step 3: Data Merging ---
    print("\n[3/6] Merging Branch Predictions...")
    # Merge semantic predictions into the lexical dataframes based on essay_id
    # Note: Since we used the same deterministic subset and seed, rows align,
    # but merging on ID is safer.

    # Helper to merge
    def merge_preds(df_base, df_new, pred_col):
        # Select only ID and prediction
        subset = df_new[["essay_id", pred_col]].copy()
        merged = df_base.merge(subset, on="essay_id", how="left")
        return merged

    df_train = merge_preds(df_train_lex, df_train_sem, "semantic_pred")
    df_val = merge_preds(df_val_lex, df_val_sem, "semantic_pred")
    df_test = merge_preds(df_test_lex, df_test_sem, "semantic_pred")

    # Fill NaNs if any (should not occur in standard flow)
    if df_train["semantic_pred"].isna().any():
        print("Warning: Found NaNs in semantic predictions. Imputing with mean.")
        mean_val = df_train["semantic_pred"].mean()
        df_train["semantic_pred"] = df_train["semantic_pred"].fillna(mean_val)
        df_val["semantic_pred"] = df_val["semantic_pred"].fillna(mean_val)
        df_test["semantic_pred"] = df_test["semantic_pred"].fillna(mean_val)

    # --- Step 4: Meta-Learner Training ---
    print("\n[4/6] Training Meta-Learner (Stacking)...")

    # Features: Predictions from both branches
    features = ["ridge_pred", "semantic_pred"]

    X_train = df_train[features].values
    y_train = df_train["score"].values

    X_val = df_val[features].values
    y_val = df_val["score"].values

    X_test = df_test[features].values

    # Train Ridge Meta-Regressor
    meta_model = Ridge(alpha=1.0, random_state=Config.SEED)
    meta_model.fit(X_train, y_train)

    print(
        f"Meta-Learner Weights -> Lexical: {meta_model.coef_[0]:.4f}, Semantic: {meta_model.coef_[1]:.4f}"
    )

    # Predict continuous scores
    val_preds_raw = meta_model.predict(X_val)
    test_preds_raw = meta_model.predict(X_test)

    # --- Step 5: Threshold Optimization & Validation ---
    print("\n[5/6] Optimizing Thresholds & Validating...")

    # Find optimal thresholds on validation set
    best_thresholds = optimize_thresholds(y_val, val_preds_raw)

    # Apply thresholds
    val_preds_final = apply_thresholds(val_preds_raw, best_thresholds)

    # Compute Metric
    final_metric = compute_qwk(y_val, val_preds_final)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    residuals = np.abs(y_val - val_preds_final)
    # Calculate word count for validation set
    word_counts = df_val["full_text"].astype(str).apply(lambda x: len(x.split()))

    # Correlation
    corr = np.corrcoef(residuals, word_counts)[0, 1]
    print(f"Correlation between Error Magnitude and Word Count: {corr:.6f}")

    # Bias check
    bias = np.mean(val_preds_final - y_val)
    print(f"Mean Prediction Bias: {bias:.6f}")

    # --- Step 6: Submission ---
    print("\n[6/6] Checking Submission Criteria...")
    TARGET_METRIC = 0.8307992749024942

    if final_metric > TARGET_METRIC:
        print(
            f"Metric ({final_metric:.6f}) > Target ({TARGET_METRIC:.6f}). Generating submission..."
        )

        # Apply thresholds to test predictions
        test_preds_final = apply_thresholds(test_preds_raw, best_thresholds)

        # Save
        generate_submission(
            df_test["essay_id"].values, test_preds_final, Config.SUBMISSION_PATH
        )
    else:
        print(
            f"Metric ({final_metric:.6f}) <= Target ({TARGET_METRIC:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
