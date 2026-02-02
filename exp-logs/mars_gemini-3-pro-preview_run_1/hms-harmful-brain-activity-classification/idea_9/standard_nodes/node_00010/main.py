import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, load_checkpoint, kl_divergence_score
from library.dataset import EEGDataset
from library.model import TimeRelativeTransformer
from library.engine import train_loop, generate_submission


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # Adjust Config for Fast Baseline
    # Limit epochs and training data size for quick execution within time limits
    Config.EPOCHS = 3
    TRAIN_SAMPLE_SIZE = 4000  # Subsample training data for speed

    # 2. Data Loading
    print("Initializing Datasets...")

    # Train Dataset (Subsampled)
    train_dataset = EEGDataset(
        mode="train", load_cached_data=True, sample_size=TRAIN_SAMPLE_SIZE
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Dataset (Full)
    val_dataset = EEGDataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Test Dataset (Full)
    test_dataset = EEGDataset(mode="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # 3. Model Initialization
    print("Initializing Model...")
    model = TimeRelativeTransformer(Config)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 4. Training
    print("Starting Training...")
    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        checkpoint_dir=Config.CHECKPOINT_DIR,
        patience=2,  # Strict patience for fast baseline
    )

    # 5. Validation & Metric Calculation
    print("Loading best model for validation...")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    load_checkpoint(best_model_path, model, device=device)
    model.eval()

    val_preds = []
    val_targets = []
    val_indices = []  # To map back to metadata

    print("Running Inference on Validation Set...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            eeg = batch["eeg"].to(device, non_blocking=True)
            spec = batch["spec"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            preds = model(eeg, spec)

            val_preds.append(preds.cpu())
            val_targets.append(targets.cpu())

            # We track indices implicitly by order since shuffle=False
            # But let's just use the length

    val_preds = torch.cat(val_preds, dim=0)
    val_targets = torch.cat(val_targets, dim=0)

    # Compute Metric
    final_metric = kl_divergence_score(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate KL divergence per sample
    # KL(P || Q) = sum(P * log(P/Q))
    # Clip Q to avoid log(0)
    epsilon = 1e-7
    Q = torch.clamp(val_preds, min=epsilon, max=1.0)
    P = val_targets

    # Calculate element-wise KL terms
    # Note: P * log(P/Q) = P * (log P - log Q)
    # Handle P=0 case: lim x->0 x log x = 0
    log_P = torch.log(torch.clamp(P, min=epsilon))
    log_Q = torch.log(Q)

    kl_per_sample = torch.sum(P * (log_P - log_Q), dim=1).numpy()

    # Load Validation Metadata to correlate
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure lengths match
    if len(val_df) != len(kl_per_sample):
        print(
            f"Warning: Metadata length ({len(val_df)}) != Predictions length ({len(kl_per_sample)})"
        )
        # Truncate to match (should not happen if loaders are correct)
        min_len = min(len(val_df), len(kl_per_sample))
        val_df = val_df.iloc[:min_len]
        kl_per_sample = kl_per_sample[:min_len]

    val_df["error"] = kl_per_sample

    # Correlation Analysis
    features_to_check = [
        "eeg_label_offset_seconds",
        "spectrogram_label_offset_seconds",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    print("Correlation between Error (KL) and Metadata features:")
    for feat in features_to_check:
        if feat in val_df.columns:
            # Drop NaNs for correlation
            valid_data = val_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["error"])
                print(f"  {feat}: {corr:.4f}")

    # 7. Conditional Submission
    THRESHOLD = 0.7327804565429688

    if final_metric < THRESHOLD:
        print(
            f"\nValidation Metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(
            model=model,
            test_loader=test_loader,
            device=device,
            output_path=Config.SUBMISSION_PATH,
        )
    else:
        print(
            f"\nValidation Metric ({final_metric}) did not beat threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
