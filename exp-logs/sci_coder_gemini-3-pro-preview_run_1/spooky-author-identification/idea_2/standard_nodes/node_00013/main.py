import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch

# Import provided library modules
from library.config import Config
from library.utils import set_seed, compute_log_loss, format_submission
from library.data_loader import load_data
from library.linear_branch import run_linear_branch
from library.transformer_branch import run_transformer_branch
from library.ensemble import run_ensemble

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    # 1. Setup and Configuration
    Config.setup()
    set_seed(Config.SEED)

    # Speed optimization for fast baseline: Reduce epochs
    Config.EPOCHS = 3

    # 2. Run Linear Branch (Stylometric)
    # This handles data loading, TF-IDF generation, training/loading, and inference
    val_probs_lin, test_probs_lin, y_val_lin = run_linear_branch(load_cached_data=True)

    # 3. Run Transformer Branch (Contextual)
    # This handles tokenization, training/loading, and inference
    val_probs_trans, test_probs_trans, y_val_trans = run_transformer_branch(
        load_cached_data=True
    )

    # Sanity check to ensure validation labels align
    if not np.array_equal(y_val_lin, y_val_trans):
        raise ValueError("Validation labels mismatch between branches.")
    y_val = y_val_lin

    # 4. Ensemble Optimization
    # Load text data for meta-features
    df_val = load_data("val")
    df_test = load_data("test")

    # Run Stacking Ensemble
    val_preds_final, test_preds_final = run_ensemble(
        val_probs_lin,
        test_probs_lin,
        val_probs_trans,
        test_probs_trans,
        y_val,
        df_val["text"],
        df_test["text"],
    )

    # 5. Validation Assessment

    # Compute Final Metric
    final_metric = compute_log_loss(y_val, val_preds_final, labels=[0, 1, 2])

    # Required Output Format
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Load validation text to analyze features
    df_val = load_data("val")
    # Feature: Text Length (Character count)
    text_lengths = df_val["text"].fillna("").astype(str).apply(len).values

    # Calculate Error Magnitude: 1.0 - Probability assigned to the true class
    # y_val contains indices [0, 1, 2]
    # val_preds_final is shape (N, 3)
    # We select the probability of the correct class for each sample
    probs_true_class = val_preds_final[np.arange(len(y_val)), y_val]
    error_magnitude = 1.0 - probs_true_class

    # Calculate Correlation
    correlation = np.corrcoef(text_lengths, error_magnitude)[0, 1]
    print(f"Correlation between Text Length and Error Magnitude: {correlation:.8f}")

    # Additional insight: Average error for short vs long texts
    median_len = np.median(text_lengths)
    avg_err_short = error_magnitude[text_lengths <= median_len].mean()
    avg_err_long = error_magnitude[text_lengths > median_len].mean()
    print(f"Avg Error (Short <= {median_len} chars): {avg_err_short:.4f}")
    print(f"Avg Error (Long > {median_len} chars): {avg_err_long:.4f}")

    # 7. Submission Generation
    THRESHOLD = 0.4535006610905359

    if final_metric < THRESHOLD:
        print("\nMetric meets threshold. Generating submission...")

        # Load Test IDs
        test_ids = df_test["id"].values

        # Format Submission
        submission_df = format_submission(
            test_ids, test_preds_final, columns=["EAP", "HPL", "MWS"]
        )

        # Save to file
        save_path = Config.SUBMISSION_FILE_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
