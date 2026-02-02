import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import set_seed
from library.train import run_training
from library.predict import generate_submission
from library.dataset import ContrailDataset, get_transforms
from library.model import ProgressiveConvNeXtUNet


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing run on {device}...")

    # 2. Data Subsetting for Fast Baseline Training
    # Load full training metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Train metadata not found at {Config.TRAIN_METADATA_PATH}"
        )

    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Sample 4000 records (~20%) to keep training time short
    # This ensures we finish within the 38-minute limit
    subset_size = 4000
    if len(full_train_df) > subset_size:
        print(f"Subsetting training data to {subset_size} samples for speed.")
        train_subset_df = full_train_df.sample(
            n=subset_size, random_state=Config.SEED
        ).reset_index(drop=True)
    else:
        train_subset_df = full_train_df

    # Save subset to working directory
    subset_path = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    train_subset_df.to_csv(subset_path, index=False)

    # Override Config path to point to the subset
    # This affects library.train.run_training
    Config.TRAIN_METADATA_PATH = subset_path

    # 3. Training
    # Run for 3 epochs to get a decent baseline quickly
    print("Starting training...")
    run_training(epochs=3, batch_size=Config.BATCH_SIZE, debug=False)

    # 4. Validation and Failure Analysis
    print("Starting validation and failure analysis...")

    # Load validation metadata (full set)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Initialize validation dataset and loader
    val_dataset = ContrailDataset(
        val_df, split="validation", transform=get_transforms("validation")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load the best model
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            "Best model checkpoint not found. Training may have failed."
        )

    model = ProgressiveConvNeXtUNet().to(device)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Metrics accumulators
    total_intersection = 0.0
    total_union = 0.0

    # Failure analysis storage
    sample_ids = []
    sample_errors = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            record_ids = batch["record_id"]

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > Config.THRESHOLD).float()

            # --- Global Dice Calculation ---
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            total_intersection += (preds_flat * masks_flat).sum().item()
            total_union += preds_flat.sum().item() + masks_flat.sum().item()

            # --- Failure Analysis (Per Sample) ---
            # Iterate through batch to calculate per-sample error
            batch_size_curr = images.size(0)
            for i in range(batch_size_curr):
                p = preds[i].view(-1)
                m = masks[i].view(-1)

                inter = (p * m).sum().item()
                union = p.sum().item() + m.sum().item()

                # Dice per sample
                dice = (2.0 * inter) / (union + 1e-6)

                # Error magnitude (1 - Dice)
                error = 1.0 - dice

                sample_ids.append(str(record_ids[i]))
                sample_errors.append(error)

    # Compute Final Global Dice
    epsilon = 1e-6
    global_dice = (2.0 * total_intersection) / (total_union + epsilon)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {global_dice}")

    # --- Failure Analysis Correlation ---
    # Create DataFrame for errors
    error_df = pd.DataFrame({"record_id": sample_ids, "error": sample_errors})

    # Merge with metadata to get features
    # Ensure record_id is string in val_df
    val_df["record_id"] = val_df["record_id"].astype(str)
    analysis_df = val_df.merge(error_df, on="record_id", how="inner")

    print("Failure Analysis - Correlation of Error with Metadata Features:")
    features = ["timestamp", "row_min", "col_min"]
    for feat in features:
        if feat in analysis_df.columns:
            corr = analysis_df[feat].corr(analysis_df["error"])
            print(f"Correlation with {feat}: {corr}")

    # 5. Submission
    threshold = 0.5910660985501295
    if global_dice > threshold:
        print(
            f"Metric ({global_dice:.6f}) exceeds threshold ({threshold}). Generating submission..."
        )
        generate_submission(
            checkpoint_path=Config.BEST_MODEL_PATH,
            batch_size=Config.BATCH_SIZE,
            output_path=Config.SUBMISSION_PATH,
            device_name=Config.DEVICE,
        )
    else:
        print(
            f"Metric ({global_dice:.6f}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
