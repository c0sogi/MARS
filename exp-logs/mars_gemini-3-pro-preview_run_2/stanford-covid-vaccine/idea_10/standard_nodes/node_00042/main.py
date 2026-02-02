import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr

# Import from provided library files
from library.config import (
    DEVICE,
    WORKING_DIR,
    METADATA_DIR,
    ALL_TARGETS,
    SCORED_TARGETS,
    SEQ_LEN,
    PRED_LEN,
    BATCH_SIZE,
    LR,
    PATIENCE,
    SEED,
)
from library.data import get_loaders
from library.model import StagedInteractiveDenseNet
from library.utils import set_seed, GlobalMCRMSE
from library.engine import train_fn, eval_fn, predict_fn

# Constants
# Override EPOCHS for fast baseline execution as requested
EPOCHS = 25
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
THRESHOLD = 0.5421870350837708

# Scored columns indices
SCORED_INDICES = [i for i, t in enumerate(ALL_TARGETS) if t in SCORED_TARGETS]
SEQ_SCORED = 68


def get_val_predictions(model, loader, device):
    """
    Runs inference on the validation loader to get raw predictions and targets
    for failure analysis.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, partners, targets in loader:
            features = features.to(device)
            partners = partners.to(device)

            outputs = model(features, partners)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


def failure_analysis(preds, targets, metadata_df):
    """
    Analyzes error correlations with metadata features.
    """
    print("\n==== Failure Analysis ====")

    # 1. Calculate RMSE per sample (on scored positions/columns only)
    # preds/targets shape: (N, 107, 5)
    p = preds[:, :SEQ_SCORED, SCORED_INDICES]
    t = targets[:, :SEQ_SCORED, SCORED_INDICES]

    # MSE per sample: mean over (SeqLen * NumTargets)
    # Shape: (N,)
    mse_per_sample = np.mean((p - t) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 2. Correlate with metadata
    # Ensure metadata aligns with validation set order.
    # The loader is shuffle=False, and metadata/val.csv should match.

    if len(metadata_df) != len(rmse_per_sample):
        print(
            f"Warning: Metadata length ({len(metadata_df)}) != Predictions length ({len(rmse_per_sample)})"
        )
        return

    # Features to analyze
    features = ["signal_to_noise", "mean_reactivity", "SN_filter"]

    # Add error to dataframe for easy correlation
    analysis_df = metadata_df.copy()
    analysis_df["rmse"] = rmse_per_sample

    for feat in features:
        if feat in analysis_df.columns:
            # Handle potential NaNs
            valid_df = analysis_df[[feat, "rmse"]].dropna()
            if len(valid_df) > 0:
                corr, _ = pearsonr(valid_df[feat], valid_df["rmse"])
                print(f"Correlation between Error (RMSE) and {feat}: {corr:.4f}")
            else:
                print(f"Could not compute correlation for {feat} (no valid data).")
        else:
            print(f"Feature {feat} not found in metadata.")


def main():
    # 1. Setup
    set_seed(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Running on Device: {DEVICE}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Load validation metadata for failure analysis
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    val_metadata = pd.read_csv(val_csv_path)

    # 3. Model Initialization
    model = StagedInteractiveDenseNet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # 4. Training Loop
    best_score = float("inf")
    early_stop_count = 0

    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, DEVICE)
        val_score = eval_fn(model, val_loader, DEVICE)

        scheduler.step(val_score)

        # Simple logging
        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f}"
        )

        if val_score < best_score:
            best_score = val_score
            early_stop_count = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_count += 1

        if early_stop_count >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 5. Final Evaluation & Metric
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    # Re-compute metric on full validation set to be precise and print required format
    # Using GlobalMCRMSE logic via eval_fn or manual calculation
    final_metric = eval_fn(model, val_loader, DEVICE)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    val_preds, val_targets = get_val_predictions(model, val_loader, DEVICE)
    failure_analysis(val_preds, val_targets, val_metadata)

    # 7. Submission
    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} is below threshold {THRESHOLD}. Generating submission..."
        )

        # Inference on Test Set
        test_preds, test_ids = predict_fn(model, test_loader, DEVICE)

        # Format Submission
        submission_data = []
        for i, sample_id in enumerate(test_ids):
            sample_preds = test_preds[i]  # (107, 5)

            for seqpos in range(PRED_LEN):
                row_id = f"{sample_id}_{seqpos}"
                vals = sample_preds[seqpos]

                row_data = {
                    "id_seqpos": row_id,
                    "reactivity": vals[0],
                    "deg_Mg_pH10": vals[1],
                    "deg_pH10": vals[2],
                    "deg_Mg_50C": vals[3],
                    "deg_50C": vals[4],
                }
                submission_data.append(row_data)

        submission_df = pd.DataFrame(submission_data)

        # Ensure column order
        cols = ["id_seqpos"] + ALL_TARGETS
        submission_df = submission_df[cols]

        # Save
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission saved to {SUBMISSION_FILE}")

    else:
        print(
            f"Metric {final_metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
