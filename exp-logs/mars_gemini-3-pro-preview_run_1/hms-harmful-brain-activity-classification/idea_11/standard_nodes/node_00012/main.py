import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, kl_divergence_score
from library.data import BrainDataset
from library.models import BidirectionalFusionNet
from library.train import train_one_epoch


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for Fast Baseline Execution
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 32
    FAST_TRAIN_SAMPLES = 15000  # Limit training data for speed

    # Setup environment
    seed_everything(Config.SEED)
    Config.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Subsample training data for fast baseline
    if len(train_df) > FAST_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(train_df)} to {FAST_TRAIN_SAMPLES} samples."
        )
        train_df = train_df.sample(
            n=FAST_TRAIN_SAMPLES, random_state=Config.SEED
        ).reset_index(drop=True)

    # Create Datasets
    # Config.LOAD_CACHED_DATA is True by default in library.config
    train_ds = BrainDataset(train_df, Config, mode="train", augment=True)
    val_ds = BrainDataset(val_df, Config, mode="val", augment=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing BidirectionalFusionNet...")
    model = BidirectionalFusionNet(Config).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # reuse train_one_epoch from library
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

    # -------------------------------------------------------------------------
    # 5. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("Performing final validation...")
    model.eval()
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            eeg = batch["eeg"].to(device)
            spec = batch["spec"].to(device)
            target = batch["target"].to(device)

            logits = model(eeg, spec)
            probs = F.softmax(logits, dim=1)

            val_probs.append(probs.cpu().numpy())
            val_targets.append(target.cpu().numpy())

    val_probs = np.concatenate(val_probs, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Final Metric
    final_metric = kl_divergence_score(val_targets, val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Calculate KL divergence per sample (Error Magnitude)
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)
    val_targets_clipped = np.clip(val_targets, epsilon, 1 - epsilon)

    # KL = sum(p * (log(p) - log(q)))
    kl_per_sample = np.sum(
        val_targets_clipped * (np.log(val_targets_clipped) - np.log(val_probs_clipped)),
        axis=1,
    )

    # Correlation with metadata features
    features = ["eeg_label_offset_seconds", "spectrogram_label_offset_seconds"]
    for feature in features:
        if feature in val_df.columns:
            # Ensure no NaNs in feature
            feat_values = val_df[feature].fillna(0).values
            correlation = np.corrcoef(feat_values, kl_per_sample)[0, 1]
            print(f"Correlation between Error (KL) and {feature}: {correlation:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.6822116374969482

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.4f}) < Threshold ({THRESHOLD:.4f}). Generating submission..."
        )

        if os.path.exists(Config.TEST_CSV):
            test_df = pd.read_csv(Config.TEST_CSV)
            test_ds = BrainDataset(test_df, Config, mode="test", augment=False)

            # Check if test set is not empty
            if len(test_ds) > 0:
                test_loader = DataLoader(
                    test_ds,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=Config.NUM_WORKERS,
                    pin_memory=True,
                )

                test_probs = []
                test_ids = []

                with torch.no_grad():
                    for batch in test_loader:
                        eeg = batch["eeg"].to(device)
                        spec = batch["spec"].to(device)
                        ids = batch["eeg_id"]

                        logits = model(eeg, spec)
                        probs = F.softmax(logits, dim=1)

                        test_probs.append(probs.cpu().numpy())
                        test_ids.extend(ids.numpy())

                test_probs = np.concatenate(test_probs, axis=0)

                # Create Submission DataFrame
                sub_df = pd.DataFrame({"eeg_id": test_ids})

                # Add vote columns
                vote_cols = [f"{c}_vote" for c in Config.CLASS_NAMES]
                for i, col in enumerate(vote_cols):
                    sub_df[col] = test_probs[:, i]

                # Save
                os.makedirs(os.path.dirname(Config.SUBMISSION_CSV), exist_ok=True)
                sub_df.to_csv(Config.SUBMISSION_CSV, index=False)
                print(f"Submission saved to {Config.SUBMISSION_CSV}")
            else:
                print("Test dataset is empty.")
        else:
            print(f"Test metadata not found at {Config.TEST_CSV}")
    else:
        print(
            f"\nMetric ({final_metric:.4f}) >= Threshold ({THRESHOLD:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
