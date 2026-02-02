import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Import from the provided library files
from library.config import SEED, VAL_METADATA_PATH, SUBMISSION_DIR, WORKING_DIR
from library.utils import seed_everything, kendall_tau_metric
from library.training_engine import train_sparse_model, train_dense_model
from library.inference_engine import (
    predict_ranks,
    anchor_sort,
    generate_submission_file,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    print("Initializing Context-Aware Hybrid Anchor Regression Pipeline...")
    seed_everything(SEED)

    # 2. Training Phase
    # The training functions handle loading data, fitting models, and saving artifacts to disk.
    # We use load_cached_data=True to leverage any pre-processed Parquet files.

    print("\n=== Phase 1: Training Sparse Stream (Ridge) ===")
    # Train Ridge Regression on TF-IDF features
    train_sparse_model(load_cached_data=True)

    print("\n=== Phase 2: Training Dense Stream (Transformer) ===")
    # Train Transformer Regressor (CodeBERT) with Structural Context
    train_dense_model(load_cached_data=True)

    # 3. Validation Phase
    print("\n=== Phase 3: Validation & Metric Calculation ===")

    # Generate predictions for the validation set using the trained ensemble
    # predict_ranks loads the models from disk and returns a DataFrame with 'pred_rank'
    df_val_preds = predict_ranks(partition="val", load_cached_data=True)

    # Reconstruct the notebook cell orders using Anchor-Based Sorting
    # This interleaves predicted markdown ranks with fixed code cell ranks
    df_val_sorted = anchor_sort(df_val_preds)

    # Convert predictions to dictionary format {id: [cell_id, ...]} for metric calculation
    val_preds_dict = {}
    for _, row in df_val_sorted.iterrows():
        val_preds_dict[row["id"]] = row["cell_order"].split()

    # Load Ground Truth from metadata
    df_val_meta = pd.read_csv(VAL_METADATA_PATH)

    # Calculate Kendall Tau Metric
    kt_score = kendall_tau_metric(df_val_meta, val_preds_dict)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {kt_score}")

    # 4. Failure Analysis
    print("\n=== Phase 4: Failure Analysis ===")

    # Calculate absolute error
    # Note: 'rank' in df_val_preds is the ground truth normalized rank
    df_val_preds["error"] = np.abs(df_val_preds["rank"] - df_val_preds["pred_rank"])

    # Extract features for correlation analysis
    df_val_preds["text_len"] = df_val_preds["text"].str.len()
    df_val_preds["context_len"] = df_val_preds["context"].str.len()

    # Calculate correlations
    correlations = {
        "text_len": df_val_preds["error"].corr(df_val_preds["text_len"]),
        "context_len": df_val_preds["error"].corr(df_val_preds["context_len"]),
        "true_rank": df_val_preds["error"].corr(df_val_preds["rank"]),
    }

    print("Correlation between Error Magnitude and Features:")
    for feature, corr in correlations.items():
        print(f"  {feature}: {corr:.4f}")

    # 5. Submission Phase
    print("\n=== Phase 5: Submission Generation ===")

    THRESHOLD = 0.7453269937267968

    if kt_score > THRESHOLD:
        print(
            f"Validation score ({kt_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission_file()
    else:
        print(
            f"Validation score ({kt_score}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
