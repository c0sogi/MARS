import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Import from provided library files
from library.config import Config
from library.utils import set_seed, compute_mcc
from library.train import train_model, generate_submission, validate
from library.dataset import get_dataloader


def perform_failure_analysis(y_true, y_probs):
    """
    Analyzes model errors on the validation set by correlating error magnitude
    with input features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate error magnitude (absolute difference)
    # y_true is 0 or 1, y_probs is [0, 1]
    errors = np.abs(y_true.flatten() - y_probs.flatten())

    # Load validation features
    # We use the cached parquet file which corresponds exactly to the validation set
    # used in the dataloader (shuffle=False).
    if not os.path.exists(Config.CACHE_VAL_X):
        print("Validation cache not found. Skipping detailed feature correlation.")
        return

    print("Loading validation features for analysis...")
    df_val_X = pd.read_parquet(Config.CACHE_VAL_X)

    if len(df_val_X) != len(errors):
        print(
            f"Shape mismatch: Features {len(df_val_X)}, Errors {len(errors)}. Skipping analysis."
        )
        return

    # Compute correlations
    # We calculate correlation between the Error vector and each Feature column
    print("Computing feature correlations with error...")

    # Using pandas corrwith for efficiency
    # Create a Series for errors with the same index
    error_series = pd.Series(errors, index=df_val_X.index)

    correlations = df_val_X.corrwith(error_series)

    # Sort by absolute correlation
    correlations_abs = correlations.abs().sort_values(ascending=False)

    print("\nTop 10 Features associated with Error (High Correlation):")
    print(correlations.loc[correlations_abs.index[:10]].to_string())

    print("\nAnalysis Complete.")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Train Model
    # train_model handles loading data, training, early stopping, and threshold optimization.
    # It returns the model with the best validation weights loaded.
    print("\n=== Starting Training Pipeline ===")
    model, best_threshold = train_model(
        num_epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        debug=False,
    )

    # 3. Validation Assessment
    print("\n=== Final Validation Assessment ===")
    # Re-run validation inference to get raw probabilities for analysis
    criterion = nn.BCELoss()
    val_loader = get_dataloader(
        "validation", batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Ensure model is in eval mode
    model.eval()

    # Get predictions
    _, y_true, y_probs = validate(model, val_loader, criterion, device)

    # Apply threshold
    y_preds = (y_probs >= best_threshold).astype(int)

    # Compute Metric
    final_mcc = compute_mcc(y_true, y_preds)

    # PRINT REQUIRED METRIC FORMAT
    print(f"Final Validation Metric: {final_mcc}")

    # 4. Failure Analysis
    perform_failure_analysis(y_true, y_probs)

    # 5. Submission
    baseline_score = 0.6190873081343531

    if final_mcc > baseline_score:
        print(
            f"\nValidation Score ({final_mcc}) > Baseline ({baseline_score}). Generating submission..."
        )
        generate_submission(
            model, best_threshold, batch_size=Config.BATCH_SIZE, load_cached_data=True
        )
    else:
        print(
            f"\nValidation Score ({final_mcc}) <= Baseline ({baseline_score}). Submission skipped."
        )


if __name__ == "__main__":
    main()
