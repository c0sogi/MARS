import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")

# Import provided library components
from library.config import Config
from library.utils import set_seed, GlobalDiceTracker
from library.dataset import ContrailDataset, get_transforms
from library.model import CascadedUNet
from library.train import fit
from library.predict import generate_submission


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    # Set reproducible seeds
    set_seed(Config.SEED)

    # Optimize Configuration for the environment (12 vCPUs, A100 GPU)
    # Modifying class attributes directly to propagate changes to library modules
    Config.NUM_WORKERS = 8
    Config.BATCH_SIZE = 64

    # Training parameters for a fast baseline
    # 5 epochs on the full dataset is sufficient for convergence
    # and fits well within the 2-hour time limit on an A100.
    FAST_EPOCHS = 5

    print(
        f"Starting Fast Baseline Run (Epochs: {FAST_EPOCHS}, Batch: {Config.BATCH_SIZE})..."
    )

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    # fit() handles the training loop and saves the best model to Config.BEST_MODEL_PATH
    fit(epochs=FAST_EPOCHS, batch_size=Config.BATCH_SIZE, debug=False)

    # ---------------------------------------------------------
    # 3. Validation & Failure Analysis
    # ---------------------------------------------------------
    print("Starting Validation and Failure Analysis...")

    device = Config.DEVICE

    # Load validation metadata
    if not os.path.exists(Config.VAL_METADATA_PATH):
        print(f"Error: Validation metadata not found at {Config.VAL_METADATA_PATH}")
        return

    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Initialize Validation Dataset and Loader
    val_dataset = ContrailDataset(
        val_df, split="validation", transform=get_transforms("validation")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Must be False to align predictions with DataFrame
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load the Best Model
    model = CascadedUNet().to(device)
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("Error: Best model checkpoint not found.")
        return

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Trackers
    global_tracker = GlobalDiceTracker()
    sample_errors = []

    # Inference Loop
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Forward pass (Model returns tuple, we want Stage 2 logits)
            _, logits = model(images)
            probs = torch.sigmoid(logits)

            # Update Global Dice Tracker (Metric Calculation)
            global_tracker.update(probs, masks)

            # Calculate per-sample error for Failure Analysis
            # Error = 1.0 - Dice
            preds_bin = (probs > 0.5).float()

            batch_size_curr = images.size(0)
            for j in range(batch_size_curr):
                p = preds_bin[j].view(-1)
                t = masks[j].view(-1)

                intersection = (p * t).sum().item()
                union = p.sum().item() + t.sum().item()

                # Handle empty-empty case (Dice = 1.0)
                if union == 0:
                    dice = 1.0
                else:
                    dice = (2.0 * intersection) / union

                sample_errors.append(1.0 - dice)

    # Compute and Print Final Metric
    final_metric = global_tracker.compute()
    print(f"Final Validation Metric: {final_metric}")

    # Perform Failure Analysis
    # Correlate error magnitude with metadata features
    if len(sample_errors) == len(val_df):
        val_df["error"] = sample_errors

        # Feature Engineering: Extract Hour from Timestamp
        val_df["datetime"] = pd.to_datetime(val_df["timestamp"], unit="s")
        val_df["hour"] = val_df["datetime"].dt.hour

        features_to_analyze = ["timestamp", "hour", "row_min", "col_min"]

        print("\nFailure Analysis - Correlation with Error (1 - Dice):")
        for feat in features_to_analyze:
            if feat in val_df.columns:
                # Compute correlation
                corr = val_df[feat].corr(val_df["error"])
                print(f"  {feat}: {corr:.4f}")
    else:
        print(
            f"Warning: Mismatch between validation samples ({len(sample_errors)}) and metadata rows ({len(val_df)}). Skipping correlation analysis."
        )

    # ---------------------------------------------------------
    # 4. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.5676456935477064

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) exceeds threshold ({THRESHOLD:.6f}). Generating submission..."
        )
        # generate_submission uses Config.BATCH_SIZE and Config.BEST_MODEL_PATH
        generate_submission(debug=False)
    else:
        print(
            f"\nMetric ({final_metric:.6f}) does not exceed threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
