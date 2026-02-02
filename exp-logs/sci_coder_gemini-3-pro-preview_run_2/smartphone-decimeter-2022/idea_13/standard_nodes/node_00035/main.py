import os
import sys
import numpy as np
import pandas as pd
import torch
import random

# Import from provided libraries
from library.config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    CONTEXT_FEATURES,
    WORKING_DIR,
    SUBMISSION_DIR,
)
from library.dataset import get_dataloader
from library.model import MultiScaleKinematicCNN, generate_submission
from library.engine import fit


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_competition_metric(df_results):
    """
    Calculates the mean of the 50th and 95th percentile distance errors.
    df_results must contain 'tripId' and 'error' columns.
    """
    # Calculate 50th and 95th percentiles for each phone (tripId)
    grouped = df_results.groupby("tripId")["error"]
    p50 = grouped.quantile(0.50)
    p95 = grouped.quantile(0.95)

    # Average the 50th and 95th percentiles for each phone
    phone_scores = (p50 + p95) / 2

    # Calculate the mean across all phones
    final_score = phone_scores.mean()
    return final_score


def run_failure_analysis(errors, context_features):
    """
    Correlates prediction errors with environmental context features.
    """
    print("\n=== Failure Analysis ===")

    # Create a DataFrame for analysis
    df_analysis = pd.DataFrame(context_features, columns=CONTEXT_FEATURES)
    df_analysis["error"] = errors

    # Compute correlations
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False)
    )

    print("Correlation between Error Magnitude and Context Features:")
    print(correlations)
    print("========================\n")


def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Use cached data if available to speed up
    train_loader = get_dataloader(
        "train", batch_size=BATCH_SIZE, shuffle=True, load_cached_data=True
    )
    val_loader = get_dataloader(
        "validation", batch_size=BATCH_SIZE, shuffle=False, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = MultiScaleKinematicCNN().to(device)

    # 4. Training
    # Limit epochs for fast baseline if needed, but using config default (30) is fine given early stopping
    print("Starting training...")
    model = fit(model, train_loader, val_loader, device, epochs=NUM_EPOCHS, patience=5)

    # 5. Validation & Metric Calculation
    print("Evaluating on validation set...")
    model.eval()

    val_preds = []
    val_targets = []
    val_ctx = []

    with torch.no_grad():
        for batch in val_loader:
            kin_seq = batch["kinematic_sequence"].to(device)
            ctx_feats = batch["context_features"].to(device)
            targets = batch["target_residual"].to(device)

            outputs = model(kin_seq, ctx_feats)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_ctx.append(ctx_feats.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    val_ctx = np.concatenate(val_ctx, axis=0)

    # Calculate Euclidean distance error in meters
    # preds and targets are (Delta East, Delta North)
    errors = np.sqrt(np.sum((val_preds - val_targets) ** 2, axis=1))

    # Get metadata for grouping
    val_meta = val_loader.dataset.meta

    # Create results DataFrame
    df_results = pd.DataFrame({"tripId": val_meta["tripId"].values, "error": errors})

    # Compute Metric
    final_metric = calculate_competition_metric(df_results)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    run_failure_analysis(errors, val_ctx)

    # 7. Submission
    THRESHOLD = 4.256982128481356
    if final_metric < THRESHOLD:
        print(
            f"Validation score ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, device, load_cached_data=True)
    else:
        print(
            f"Validation score ({final_metric}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
