import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.data_loader import load_data
from library.nbsvm_model import get_structural_features
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

    # 3. Generate Structural Features (Dense SVD)
    print("\n=== Generating Structural Features ===")
    train_struct, val_struct, test_struct = get_structural_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 4. Train Hybrid Model (Transformer + Structural Fusion)
    print("\n=== Running Hybrid Model ===")
    val_preds, test_preds = run_transformer(
        train_df,
        val_df,
        test_df,
        train_struct,
        val_struct,
        test_struct,
        load_cached_data=True,
    )

    # 6. Final Validation Metric
    y_val = val_df["Insult"].values
    final_auc = roc_auc_score(y_val, val_preds)

    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    perform_failure_analysis(val_df, val_preds)

    # 8. Submission
    threshold = 0.9582101806239737

    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        final_test_preds = test_preds

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
