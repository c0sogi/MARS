import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import logging
from transformers import logging as transformers_logging

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.lexical_branch import run_tfidf_ridge
from library.semantic_trainer import run_semantic_training
from library.postprocessing import (
    optimize_thresholds,
    apply_thresholds,
    generate_submission,
)

# --- Setup & Configuration ---
# Suppress warnings and logs for cleaner output
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_demo():
    print("=== Starting Essay Scoring Pipeline Demo ===\n")

    # 1. Setup
    seed_everything(Config.SEED)

    # Modify Config for a fast demonstration
    # We use class attribute modification to propagate settings
    print("Configuring for fast demo execution...")
    Config.DEBUG = True
    Config.N_FOLDS = 2  # Minimum folds for cross-validation logic
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch per fold in demo
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for speed

    # Ensure working directory is clean-ish or ready
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Run Lexical Branch (TF-IDF + Ridge)
    # ---------------------------------------------------------
    print("\n[Step 1] Running Lexical Branch (TF-IDF + Ridge)...")

    # We force load_cached_data=False to demonstrate the fitting logic
    df_train_lex, df_val_lex, df_test_lex = run_tfidf_ridge(
        load_cached_data=False, debug=True
    )

    # Verification
    assert "ridge_pred" in df_train_lex.columns, "Train DF missing ridge_pred"
    assert "ridge_pred" in df_val_lex.columns, "Val DF missing ridge_pred"
    assert "ridge_pred" in df_test_lex.columns, "Test DF missing ridge_pred"

    # Check range
    assert df_val_lex["ridge_pred"].min() >= 1.0, "Predictions below 1.0 found"
    assert df_val_lex["ridge_pred"].max() <= 6.0, "Predictions above 6.0 found"

    print(f"Lexical Branch Complete. Val Shape: {df_val_lex.shape}")
    print(f"Sample Lexical Preds: {df_val_lex['ridge_pred'].head().tolist()}")

    # ---------------------------------------------------------
    # 3. Run Semantic Branch (DeBERTa)
    # ---------------------------------------------------------
    print("\n[Step 2] Running Semantic Branch (DeBERTa-v3-Large)...")
    print(
        "Note: This step involves loading a large model and may take a few minutes even on a subset."
    )

    # We run the semantic trainer. It internally handles tokenization,
    # dataset creation, and training loop.
    df_train_sem, df_val_sem, df_test_sem = run_semantic_training(
        debug=True, load_cached_data=False
    )

    # Verification
    assert "semantic_pred" in df_train_sem.columns, "Train DF missing semantic_pred"
    assert "semantic_pred" in df_val_sem.columns, "Val DF missing semantic_pred"
    assert "semantic_pred" in df_test_sem.columns, "Test DF missing semantic_pred"

    # Check model artifacts
    expected_model_path = os.path.join(Config.MODEL_OUTPUT_DIR, "deberta_fold_0.bin")
    if os.path.exists(expected_model_path):
        print(f"Verified model artifact exists: {expected_model_path}")
    else:
        # It's possible the model wasn't saved if validation loss didn't improve,
        # but with 1 epoch and init, it usually saves once.
        print(f"Warning: Model artifact not found at {expected_model_path}")

    print(f"Semantic Branch Complete. Val Shape: {df_val_sem.shape}")
    print(f"Sample Semantic Preds: {df_val_sem['semantic_pred'].head().tolist()}")

    # ---------------------------------------------------------
    # 4. Ensembling
    # ---------------------------------------------------------
    print("\n[Step 3] Ensembling Predictions...")

    # Align DataFrames by essay_id to ensure correctness
    # In this demo, they come from the same source splits, so indices should align if not shuffled differently.
    # However, merge is safer.

    # Validation Set Ensemble
    val_merged = df_val_lex[["essay_id", "score", "ridge_pred"]].merge(
        df_val_sem[["essay_id", "semantic_pred"]], on="essay_id", how="inner"
    )

    # Test Set Ensemble
    test_merged = df_test_lex[["essay_id", "ridge_pred"]].merge(
        df_test_sem[["essay_id", "semantic_pred"]], on="essay_id", how="inner"
    )

    # Weighted Average (Simple 50/50 for demo)
    w_lex = 0.4
    w_sem = 0.6

    val_merged["ensemble_pred"] = (val_merged["ridge_pred"] * w_lex) + (
        val_merged["semantic_pred"] * w_sem
    )
    test_merged["ensemble_pred"] = (test_merged["ridge_pred"] * w_lex) + (
        test_merged["semantic_pred"] * w_sem
    )

    print(f"Ensemble created. Combined {len(val_merged)} validation samples.")

    # ---------------------------------------------------------
    # 5. Post-Processing (Threshold Optimization)
    # ---------------------------------------------------------
    print("\n[Step 4] Optimizing Thresholds...")

    # Optimize thresholds on Validation set
    y_true_val = val_merged["score"].values
    y_pred_val = val_merged["ensemble_pred"].values

    best_thresholds = optimize_thresholds(y_true_val, y_pred_val)

    # Apply to Test set
    test_preds_continuous = test_merged["ensemble_pred"].values
    test_preds_int = apply_thresholds(test_preds_continuous, best_thresholds)

    print(f"Optimized Thresholds: {best_thresholds}")
    print(f"Test Predictions (Integer): {test_preds_int[:10]}")

    # ---------------------------------------------------------
    # 6. Generate Submission
    # ---------------------------------------------------------
    print("\n[Step 5] Generating Submission File...")

    submission_path = Config.SUBMISSION_PATH
    essay_ids = test_merged["essay_id"].values

    generate_submission(essay_ids, test_preds_int, submission_path)

    # Verify file
    assert os.path.exists(submission_path), "Submission file was not created."

    # Read back to check format
    sub_df = pd.read_csv(submission_path)
    assert list(sub_df.columns) == ["essay_id", "score"], "Incorrect submission columns"
    assert (
        sub_df["score"].dtype == int or sub_df["score"].dtype == np.int64
    ), "Score column is not integer"
    assert len(sub_df) == len(essay_ids), "Submission length mismatch"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
