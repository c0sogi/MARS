import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from provided library files
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import RFMHDNetwork
from library.train import run_training, generate_submission

# ==========================================
# Configuration
# ==========================================
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 5
THRESHOLD_AUC = 0.6978181818181817


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model errors on the validation set and correlates them with metadata features.
    """
    print("\n" + "=" * 40)
    print(" FAILURE ANALYSIS")
    print("=" * 40)

    # 1. Get Predictions
    model.eval()
    all_preds = []
    all_targets = []

    # Ensure no gradients are computed
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 2. Calculate Errors
    errors = np.abs(all_targets - all_preds)

    # 3. Load Metadata to extract features
    val_meta_path = os.path.join(METADATA_DIR, "val.parquet")
    if not os.path.exists(val_meta_path):
        print("Validation metadata not found. Skipping detailed metadata correlation.")
        return

    val_df = pd.read_parquet(val_meta_path)

    # Ensure alignment: val_loader (shuffle=False) should match val_df order
    if len(val_df) != len(errors):
        print(
            f"Warning: Metadata size ({len(val_df)}) does not match prediction size ({len(errors)}). Skipping correlation."
        )
        return

    # 4. Extract Features and Correlate
    # Feature: Slice counts per modality
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    correlations = {}

    print(
        f"Analyzing correlations between Absolute Error and metadata features (N={len(errors)})..."
    )

    for mod in modalities:
        col_name = f"{mod}_paths"
        # Calculate slice count for each patient
        slice_counts = (
            val_df[col_name].apply(lambda x: len(x) if x is not None else 0).values
        )

        # Calculate correlation
        if np.std(slice_counts) > 0:
            corr, _ = pearsonr(errors, slice_counts)
            correlations[f"{mod}_slice_count"] = corr
        else:
            correlations[f"{mod}_slice_count"] = 0.0

    # Feature: Target Class (Bias check)
    corr_target, _ = pearsonr(errors, all_targets)
    correlations["target_value"] = corr_target

    # Print Results
    print("-" * 30)
    print(f"{'Feature':<20} | {'Correlation with Error':<20}")
    print("-" * 30)
    for feature, corr in correlations.items():
        print(f"{feature:<20} | {corr:.4f}")
    print("-" * 30)

    # Interpretation
    max_corr_feat = max(correlations, key=lambda k: abs(correlations[k]))
    print(f"Strongest correlation: {max_corr_feat} ({correlations[max_corr_feat]:.4f})")


def main():
    # 1. Setup
    set_seed(42)
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Training
    print("\nStarting Training Pipeline...")
    # run_training handles dataloading, model init, training loop, and saving best model
    best_val_auc = run_training(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        load_cached_data=True,
        save_path=BEST_MODEL_PATH,
    )

    # 3. Validation & Metric Reporting
    print("\nLoading best model for final validation...")

    # Re-load dataloaders to get the validation set
    _, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE, load_cached_data=True)

    # Initialize model and load weights
    model = RFMHDNetwork(pretrained=False)
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    else:
        print("Error: Best model file not found!")
        return

    model.to(device)
    model.eval()

    # Compute final metric on validation set
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_targets.extend(targets.numpy().flatten())
            all_preds.extend(probs)

    final_metric = roc_auc_score(all_targets, all_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 5. Submission
    if final_metric > THRESHOLD_AUC:
        print(
            f"\nValidation Metric ({final_metric:.5f}) > Threshold ({THRESHOLD_AUC:.5f}). Generating submission..."
        )
        generate_submission(
            model_path=BEST_MODEL_PATH,
            output_path=SUBMISSION_PATH,
            batch_size=BATCH_SIZE,
            load_cached_data=True,
        )
    else:
        print(
            f"\nValidation Metric ({final_metric:.5f}) <= Threshold ({THRESHOLD_AUC:.5f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
