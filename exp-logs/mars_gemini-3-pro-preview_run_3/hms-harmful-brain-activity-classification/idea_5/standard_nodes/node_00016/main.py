import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import provided library modules
import library.config as config
import library.train as train_module
import library.inference as inference_module
import library.model as model_module
import library.dataset as dataset_module
import library.utils as utils_module


def main():
    # ==========================================
    # 1. Configuration Override for Fast Baseline
    # ==========================================
    # Adjust hyperparameters to fit within the 2-hour time limit
    # A100 GPU allows for larger batch size
    config.EPOCHS = 6
    config.BATCH_SIZE = 64
    config.NUM_WORKERS = 4

    print(f"Configuration set: Epochs={config.EPOCHS}, Batch Size={config.BATCH_SIZE}")

    # Set seeds for reproducibility
    utils_module.seed_everything(config.SEED)

    # ==========================================
    # 2. Training
    # ==========================================
    print("\nStarting Training Pipeline...")
    # Run training (debug=False to use full dataset, load_cached_data=True to speed up)
    train_module.train(debug=False, load_cached_data=True)

    # ==========================================
    # 3. Validation & Metric Calculation
    # ==========================================
    print("\nStarting Validation & Failure Analysis...")
    device = torch.device(config.DEVICE)

    # Load Validation Dataset
    val_dataset = dataset_module.HMSDataset(
        csv_file=config.VAL_META_PATH, mode="val", augment=False, load_cached_data=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model = model_module.HybridModel()
    model.to(device)

    try:
        utils_module.load_checkpoint(model, config.MODEL_PATH, device)
        print(f"Loaded checkpoint from {config.MODEL_PATH}")
    except FileNotFoundError:
        print("Error: Model checkpoint not found. Training may have failed.")
        sys.exit(1)

    model.eval()

    # Inference on Validation Set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for eeg, spec, targets in val_loader:
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            logits = model(eeg, spec)

            # Apply Log Softmax for KL Div calculation (Logits -> LogProbs)
            log_probs = F.log_softmax(logits, dim=1)

            all_preds.append(log_probs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate
    y_log_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    # Compute Global KL Divergence Metric (Batchmean)
    # KLDivLoss expects input=log_probs, target=probs
    kl_loss = F.kl_div(y_log_pred, y_true, reduction="batchmean")
    final_metric = kl_loss.item()

    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")

    # Compute per-sample loss (reduction='none') and sum across classes
    # KL(P || Q) = sum(P * log(P/Q)) = sum(P * (log P - log Q))
    # PyTorch kl_div(log_Q, P) calculates P * (log P - log Q)
    # We sum across the class dimension (dim=1) to get scalar loss per sample
    per_sample_loss = F.kl_div(y_log_pred, y_true, reduction="none").sum(dim=1).numpy()

    # Load metadata to correlate
    val_df = pd.read_csv(config.VAL_META_PATH)

    # Ensure alignment
    if len(val_df) != len(per_sample_loss):
        print(
            "Warning: Metadata length mismatch. Skipping detailed correlation analysis."
        )
    else:
        # Add loss to dataframe
        val_df["kl_loss"] = per_sample_loss

        # Features to analyze
        analysis_cols = [
            "total_votes",
            "eeg_label_offset_seconds",
            "spectogram_label_offset_seconds",
        ]

        print("Correlation between Error (KL Loss) and Metadata features:")
        for col in analysis_cols:
            if col in val_df.columns:
                # Handle NaNs just in case
                valid_mask = val_df[col].notna() & val_df["kl_loss"].notna()
                if valid_mask.sum() > 1:
                    corr, _ = pearsonr(
                        val_df.loc[valid_mask, col], val_df.loc[valid_mask, "kl_loss"]
                    )
                    print(f"  {col}: {corr:.4f}")
                else:
                    print(f"  {col}: Not enough data")
            else:
                print(f"  {col}: Column not found")

    # ==========================================
    # 5. Submission
    # ==========================================
    THRESHOLD = 1.0081

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        inference_module.predict(
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            load_cached_data=True,
            device=config.DEVICE,
        )
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
