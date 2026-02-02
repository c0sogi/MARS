import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import torch.optim as optim

# Import from provided library
from library.config import Config
from library.utils import seed_everything
from library.loss import MCRMSELoss
from library.data import get_loader
from library.model import DIN_CG_BiGRU
from library.train import train_one_epoch, validate

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates sample-wise MCRMSE and correlates it with metadata features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # Inference
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            bpps_indices = batch["bpps_indices"].to(device)
            bpps_mask = batch["bpps_mask"].to(device)
            targets = batch["targets"]  # Keep on CPU
            ids = batch["ids"]

            outputs = model(features, bpps_indices, bpps_mask)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_ids.extend(ids)

    # Concatenate
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Slice to scored length for error calculation
    preds_sliced = preds[:, : Config.PRED_LEN, :]
    targets_sliced = targets[:, : Config.PRED_LEN, :]

    # Calculate MCRMSE per sample
    # (N, 68, 5) -> (N,)
    mse_per_sample = np.mean((preds_sliced - targets_sliced) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(
        mse_per_sample
    )  # This is a simplified sample-level metric

    # Better: Calculate MCRMSE per sample exactly as metric definition
    # Root of mean squared error per column, averaged over columns
    # For a single sample, we average the RMSE of the 5 columns
    col_mse = np.mean((preds_sliced - targets_sliced) ** 2, axis=1)  # (N, 5)
    col_rmse = np.sqrt(col_mse)
    sample_mcrmse = np.mean(col_rmse, axis=1)  # (N,)

    # Load Metadata to get features
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)

    # Ensure alignment (val_loader is not shuffled)
    # The IDs in df_val should match all_ids
    # We can merge on ID to be safe
    error_df = pd.DataFrame({"id": all_ids, "error": sample_mcrmse})
    analysis_df = pd.merge(df_val, error_df, on="id")

    # Calculate Correlations
    features_to_check = ["signal_to_noise", "SN_filter", "seq_length"]
    print("Correlation between Error (MCRMSE) and Features:")
    for feat in features_to_check:
        if feat in analysis_df.columns:
            corr = analysis_df[feat].corr(analysis_df["error"])
            print(f"  {feat}: {corr:.6f}")


def generate_submission(model, device):
    """
    Generates submission.csv for the test set.
    """
    print("\n==== Generating Submission ====")
    model.eval()

    test_loader = get_loader(
        "test", batch_size=Config.BATCH_SIZE, shuffle=False, load_cached_data=True
    )

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            bpps_indices = batch["bpps_indices"].to(device)
            bpps_mask = batch["bpps_mask"].to(device)
            ids = batch["ids"]

            outputs = model(features, bpps_indices, bpps_mask)
            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate predictions: (N, 107, 5)
    preds = np.concatenate(all_preds, axis=0)

    # Prepare submission rows
    submission_rows = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(all_ids):
        sample_pred = preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_data = {
                "id_seqpos": row_id,
                "reactivity": sample_pred[seqpos, 0],
                "deg_Mg_pH10": sample_pred[seqpos, 1],
                "deg_pH10": sample_pred[seqpos, 2],
                "deg_Mg_50C": sample_pred[seqpos, 3],
                "deg_50C": sample_pred[seqpos, 4],
            }
            submission_rows.append(row_data)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Configuration Override for Fast Baseline
    Config.EPOCHS = 15
    print(f"Running Fast Baseline with EPOCHS={Config.EPOCHS}")

    # 2. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 3. Data Loading
    print("Loading Data...")
    train_loader = get_loader(
        "train", batch_size=Config.BATCH_SIZE, shuffle=True, load_cached_data=True
    )
    val_loader = get_loader(
        "val", batch_size=Config.BATCH_SIZE, shuffle=False, load_cached_data=True
    )

    # 4. Model Initialization
    print("Initializing Model...")
    model = DIN_CG_BiGRU().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = MCRMSELoss()

    # 5. Training Loop
    best_score = float("inf")

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Checkpoint
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print("-" * 30)
    print(f"Final Validation Metric: {best_score}")
    print("-" * 30)

    # 6. Load Best Model for Analysis and Submission
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(device)

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 8. Conditional Submission
    THRESHOLD = 0.5978901386
    if best_score < THRESHOLD:
        generate_submission(model, device)
    else:
        print(
            f"Validation score ({best_score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
