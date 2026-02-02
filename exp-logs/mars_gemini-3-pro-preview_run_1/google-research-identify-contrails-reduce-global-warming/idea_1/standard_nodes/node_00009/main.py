import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.train import train_model, validate, ContrailLoss
from library.inference import generate_submission
from library.dataset import ContrailDataset
from library.model import SimpleUNet
from library.utils import dice_coefficient


def run_failure_analysis(model, dataset, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error (1 - Dice) and metadata features.
    """
    print("Running Failure Analysis...")
    model.eval()

    # Create a loader that doesn't shuffle to align with dataframe
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    errors = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            # Binarize predictions for metric calculation
            preds = (outputs > Config.THRESHOLD).float()

            # Calculate per-sample error
            for i in range(images.size(0)):
                # Pass threshold=None as preds are already binary
                d = dice_coefficient(preds[i], masks[i], threshold=None)
                errors.append(1.0 - d.item())

    # Get metadata
    df = dataset.df.copy()

    # Ensure lengths match
    if len(errors) != len(df):
        n = min(len(errors), len(df))
        errors = errors[:n]
        df = df.iloc[:n]

    df["error"] = errors

    # Feature Engineering for Analysis
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df["hour"] = df["datetime"].dt.hour

    # Select numeric features for correlation
    features = ["timestamp", "hour", "row_min", "col_min", "row_size", "col_size"]
    valid_features = [f for f in features if f in df.columns]

    print("Correlation between Error (1 - Dice) and Features:")
    correlations = df[valid_features].corrwith(df["error"])
    print(correlations)


def main():
    # 1. Setup
    Config.set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Training
    print("\n=== Starting Training ===")
    # Train on full dataset (max_samples=None) for full epochs to overcome class imbalance
    train_model(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        max_samples=None,
        patience=10,
    )

    # 3. Validation on Full Hold-out Set
    print("\n=== Starting Full Validation ===")

    # Load best model
    model = SimpleUNet(in_channels=Config.NUM_BANDS, out_channels=1)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found. Training might have failed.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Load full validation set
    val_dataset = ContrailDataset(split="validation")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    criterion = ContrailLoss()

    # Calculate global metric
    _, final_dice = validate(model, val_loader, criterion, device)

    # Print required metric
    print(f"Final Validation Metric: {final_dice}")

    # 4. Failure Analysis
    print("\n=== Starting Failure Analysis ===")
    run_failure_analysis(model, val_dataset, device)

    # 5. Inference / Submission
    print("\n=== Checking Submission Criteria ===")
    threshold_score = 0.5472023931632809

    if final_dice > threshold_score:
        print(
            f"Validation Metric {final_dice:.4f} > {threshold_score:.4f}. Generating Submission..."
        )
        generate_submission(
            checkpoint_path=checkpoint_path, batch_size=Config.BATCH_SIZE, device=device
        )
    else:
        print(
            f"Validation Metric {final_dice:.4f} <= {threshold_score:.4f}. Skipping Submission."
        )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
