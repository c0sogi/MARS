import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, KL_loss
from library.data import get_dataloaders
from library.models import DualStreamModel
from library.engine import fit


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Create submission directory
    os.makedirs("./submission", exist_ok=True)

    # --- Fast Baseline Configuration Overrides ---
    # We want to train fast, but validate on the FULL validation set.
    # The provided data.py slices both if DEBUG_SAMPLE_SIZE > 0.
    # Strategy: Create a subsampled training CSV, but use full validation CSV.

    print("Preparing subsampled training data for fast baseline...")
    # Load original train csv
    full_train_df = pd.read_csv(Config.TRAIN_CSV)

    # Sample 20,000 samples for speed (approx 25% of data) to ensure run completes < 2hrs
    sub_train_df = full_train_df.sample(n=20000, random_state=Config.SEED).reset_index(
        drop=True
    )

    # Save temporary subsampled file
    sub_train_path = os.path.join(Config.WORKING_DIR, "train_subsampled.csv")
    sub_train_df.to_csv(sub_train_path, index=False)

    # Override Config
    Config.EPOCHS = 3
    Config.DEBUG_SAMPLE_SIZE = 0  # Disable internal slicing to keep Val set full
    Config.BATCH_SIZE = 32

    print(f"Training on {len(sub_train_df)} samples. Validating on full set.")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    # We pass the subsampled training path, but the original validation/test paths
    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv_path=sub_train_path,
        val_csv_path=Config.VAL_CSV,
        test_csv_path=Config.TEST_CSV,
        config=Config,
    )

    # 3. Model Initialization
    model = DualStreamModel(config=Config, pretrained=True)
    model.to(Config.DEVICE)

    # 4. Optimization
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

    # 5. Training
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        config=Config,
    )

    # 6. Validation and Failure Analysis
    print("\nStarting Validation & Failure Analysis...")

    # Load best model
    best_model_path = os.path.join(Config.get_output_dir(), "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    else:
        print("Warning: Best model not found, using current weights.")

    model.eval()

    val_probs_list = []
    val_targets_list = []

    # Inference on Validation Set
    with torch.no_grad():
        for eeg, spec, targets in val_loader:
            eeg = eeg.to(Config.DEVICE)
            spec = spec.to(Config.DEVICE)

            logits = model(eeg, spec)
            probs = torch.softmax(logits, dim=1)

            val_probs_list.append(probs.cpu().numpy())
            val_targets_list.append(targets.numpy())

    val_probs = np.concatenate(val_probs_list)
    val_targets = np.concatenate(val_targets_list)

    # Calculate Final Metric
    final_metric = KL_loss(val_probs, val_targets)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    # Compute KL per sample: sum(p * (log p - log q))
    epsilon = 1e-15
    p = np.clip(val_targets, epsilon, 1 - epsilon)
    q = np.clip(val_probs, epsilon, 1 - epsilon)
    kl_per_sample = np.sum(p * (np.log(p) - np.log(q)), axis=1)

    # Get metadata from dataset
    val_df = val_loader.dataset.df

    # Check alignment
    if len(val_df) == len(kl_per_sample):
        # EEG Offset Correlation
        if "eeg_label_offset_seconds" in val_df.columns:
            offsets = val_df["eeg_label_offset_seconds"].fillna(0).values
            corr, _ = pearsonr(offsets, kl_per_sample)
            print(f"Correlation (Error vs EEG Offset): {corr:.4f}")

        # Spectrogram Offset Correlation
        if "spectrogram_label_offset_seconds" in val_df.columns:
            offsets = val_df["spectrogram_label_offset_seconds"].fillna(0).values
            corr, _ = pearsonr(offsets, kl_per_sample)
            print(f"Correlation (Error vs Spectrogram Offset): {corr:.4f}")
    else:
        print("Error: Validation dataframe length mismatch with predictions.")

    # 7. Submission Generation
    THRESHOLD = 0.7327804565429688

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is lower than threshold ({THRESHOLD}). Generating submission..."
        )

        test_probs_list = []

        with torch.no_grad():
            for eeg, spec, _ in test_loader:
                eeg = eeg.to(Config.DEVICE)
                spec = spec.to(Config.DEVICE)

                logits = model(eeg, spec)
                probs = torch.softmax(logits, dim=1)

                test_probs_list.append(probs.cpu().numpy())

        test_probs = np.concatenate(test_probs_list)

        # Construct Submission DataFrame
        test_df = test_loader.dataset.df
        submission = pd.DataFrame()
        submission["eeg_id"] = test_df["eeg_id"]

        # Columns must be in specific order
        # 0: seizure, 1: lpd, 2: gpd, 3: lrda, 4: grda, 5: other
        cols = [
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
        ]
        for i, col in enumerate(cols):
            submission[col] = test_probs[:, i]

        # Save
        sub_path = "./submission/submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric ({final_metric}) is NOT lower than threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
