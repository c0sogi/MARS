import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, GlobalMetrics
from library.data import RNADataset
from library.model import DDFRN
from library.train import Trainer
from library.loss import MaskedMCRMSELoss


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between Error and Signal-to-Noise ratio.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    all_errors = []
    all_ids = []

    # 1. Compute per-sample error
    # We use the same masking logic as the loss/metric: first 68 bases, specific columns
    scored_indices = Config.SCORED_INDICES
    seq_scored = Config.SEQ_SCORED

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            # Forward pass (use refined prediction y2)
            _, y_pred = model(inputs, partner_indices)

            # Slice and Filter
            y_pred = y_pred[:, :seq_scored, scored_indices]
            targets = targets[:, :seq_scored, scored_indices]

            # Compute RMSE per sample (average over sequence and channels, then sqrt)
            # Shape: (Batch, Seq, Channels) -> (Batch,)
            mse_per_sample = torch.mean((y_pred - targets) ** 2, dim=(1, 2))
            rmse_per_sample = torch.sqrt(mse_per_sample)

            all_errors.extend(rmse_per_sample.cpu().numpy())
            all_ids.extend(ids)

    # 2. Load Metadata
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Create a DataFrame for analysis
    error_df = pd.DataFrame({"id": all_ids, "error": all_errors})

    # Merge with metadata to get signal_to_noise
    # Ensure we match on ID
    analysis_df = pd.merge(
        error_df, val_df[["id", "signal_to_noise", "SN_filter"]], on="id", how="left"
    )

    # 3. Calculate Correlations
    if "signal_to_noise" in analysis_df.columns:
        # Drop NaNs just in case
        valid_df = analysis_df.dropna(subset=["error", "signal_to_noise"])
        if len(valid_df) > 1:
            corr, _ = pearsonr(valid_df["error"], valid_df["signal_to_noise"])
            print(f"Correlation (Error vs Signal_to_Noise): {corr:.4f}")
        else:
            print("Not enough data for correlation analysis.")

    if "SN_filter" in analysis_df.columns:
        valid_df = analysis_df.dropna(subset=["error", "SN_filter"])
        if len(valid_df) > 1:
            corr, _ = pearsonr(valid_df["error"], valid_df["SN_filter"])
            print(f"Correlation (Error vs SN_filter):      {corr:.4f}")


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("\n==== Generating Submission ====")

    # Load Test Data
    test_dataset = RNADataset(mode="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    model.eval()
    preds = []
    ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            batch_ids = batch["id"]

            # Forward pass (y2)
            _, y_pred = model(inputs, partner_indices)

            # Move to CPU numpy
            y_pred = y_pred.cpu().numpy()  # (B, 107, 5)

            preds.append(y_pred)
            ids.extend(batch_ids)

    preds = np.concatenate(preds, axis=0)  # (N_samples, 107, 5)

    # Format for submission
    # We need to flatten: id_seqpos, val1, val2, val3, val4, val5
    submission_rows = []
    target_cols = (
        Config.TARGET_COLS
    )  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()

            row_dict = {"id_seqpos": row_id}
            for col_name, val in zip(target_cols, row_values):
                row_dict[col_name] = val

            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    # Reorder columns to match sample submission
    cols = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols]

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading Datasets...")
    train_dataset = RNADataset(mode="train", load_cached_data=True)
    val_dataset = RNADataset(mode="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = DDFRN().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    criterion = MaskedMCRMSELoss()

    # 4. Training
    trainer = Trainer(model, device, criterion, optimizer, scheduler)
    trainer.fit(
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_PATH,
    )

    # 5. Final Evaluation
    print("\n==== Final Evaluation ====")
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    metric_calculator = GlobalMetrics()

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            # Inference (y2)
            _, y2 = model(inputs, partner_indices)

            metric_calculator.update(targets, y2)

    final_metric = metric_calculator.compute()
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Submission
    THRESHOLD = 0.47142532743789534
    if final_metric < THRESHOLD:
        generate_submission(model, device)
    else:
        print(f"Validation metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
