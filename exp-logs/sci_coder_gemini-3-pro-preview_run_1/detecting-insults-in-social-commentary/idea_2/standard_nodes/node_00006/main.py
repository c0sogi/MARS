import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.data_loader import load_data
from library.nbsvm_model import run_nbsvm
from library.transformer_model import run_transformer, set_seed


def perform_failure_analysis(val_df, val_preds):
    """
    Analyzes model errors on the validation set.
    Computes correlations between error magnitude and input features.
    """
    print("\nPerforming Failure Analysis...")
    analysis_df = val_df.copy()

    # Calculate error magnitude
    y_true = analysis_df["Insult"].values
    analysis_df["pred"] = val_preds
    analysis_df["error"] = np.abs(y_true - analysis_df["pred"])

    # Generate simple features for correlation
    # Handle potential non-string values gracefully
    analysis_df["text_len"] = analysis_df["Comment"].astype(str).apply(len)
    analysis_df["word_count"] = (
        analysis_df["Comment"].astype(str).apply(lambda x: len(x.split()))
    )

    def get_caps_ratio(text):
        s = str(text)
        if len(s) == 0:
            return 0.0
        return sum(1 for c in s if c.isupper()) / len(s)

    analysis_df["caps_ratio"] = analysis_df["Comment"].apply(get_caps_ratio)

    # Calculate correlations
    features_to_check = ["text_len", "word_count", "caps_ratio"]
    correlations = (
        analysis_df[features_to_check + ["error"]].corr()["error"].drop("error")
    )

    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    return correlations


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Load Data
    print("Loading data...")
    train_df, val_df, test_df = load_data(load_cached_data=True)

    # 3. Train Structural Branch (NBSVM)
    print("\n=== Running Structural Branch (NBSVM) ===")
    val_preds_nb, test_preds_nb = run_nbsvm(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 4. Train Semantic Branch (Transformer)
    print("\n=== Running Semantic Branch (Transformer) ===")
    # Note: run_transformer handles its own device placement and evaluation mode
    val_preds_tr, test_preds_tr = run_transformer(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 5. Ensemble Optimization
    print("\n=== Optimizing Ensemble Weights ===")
    y_val = val_df["Insult"].values

    best_auc = 0.0
    best_w = 0.0

    # Simple grid search for weight w (Transformer weight)
    # P_final = w * P_tr + (1-w) * P_nb
    steps = 101
    for i in range(steps):
        w = i / (steps - 1)
        blended_preds = w * val_preds_tr + (1 - w) * val_preds_nb
        current_auc = roc_auc_score(y_val, blended_preds)

        if current_auc > best_auc:
            best_auc = current_auc
            best_w = w

    print(f"Optimal Transformer Weight: {best_w:.2f}")
    print(f"Optimal NBSVM Weight: {1 - best_w:.2f}")

    # 6. Final Validation Metric
    # Recalculate best preds to ensure precision
    final_val_preds = best_w * val_preds_tr + (1 - best_w) * val_preds_nb
    final_auc = roc_auc_score(y_val, final_val_preds)

    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    perform_failure_analysis(val_df, final_val_preds)

    # 8. Submission
    threshold = 0.8992692939244664

    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Blend test predictions
        final_test_preds = best_w * test_preds_tr + (1 - best_w) * test_preds_nb

        # Load sample submission to preserve format
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Ensure lengths match
        if len(sample_sub) != len(final_test_preds):
            print(
                f"Warning: Sample submission length ({len(sample_sub)}) matches test preds ({len(final_test_preds)})?"
            )
            # We assume the test_df order matches sample_submission as per standard competition formats
            # If lengths differ, we might need to rely on test_df index or IDs if available.
            # Based on metadata generation, test_df comes from test.csv which matches sample_submission rows.

        # Assign predictions
        # The sample submission has columns: Insult, Date, Comment.
        # We need to fill 'Insult' with probabilities (or class labels? Task says "predictions should be a number in the range [0,1]")
        sample_sub["Insult"] = final_test_preds

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sample_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({final_auc}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
