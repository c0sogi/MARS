import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import set_seed, dice_score
from library.dataset import ContrailDataset
from library.model import ResNet34UNet
from library.train import run_training
from library.predict import generate_submission


def analyze_failures(model, device):
    """
    Performs validation inference, calculates Global Dice, and analyzes failure modes
    by correlating error with metadata features.
    """
    print("\nStarting Failure Analysis on Validation Set...")

    # 1. Load Validation Data
    val_dataset = ContrailDataset(split="validation")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load metadata for correlation analysis
    meta_df = val_dataset.df.copy()

    # 2. Run Inference
    model.eval()

    intersection_sum = 0.0
    union_sum = 0.0
    per_sample_dice = []

    # Ensure no gradients for speed
    with torch.no_grad():
        current_idx = 0
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Update Global Dice accumulators
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            intersection_sum += (preds_flat * masks_flat).sum().item()
            union_sum += preds_flat.sum().item() + masks_flat.sum().item()

            # Calculate per-sample Dice for failure analysis
            batch_size = images.size(0)
            for i in range(batch_size):
                p_flat = preds[i].view(-1)
                m_flat = masks[i].view(-1)

                inter = (p_flat * m_flat).sum().item()
                union = p_flat.sum().item() + m_flat.sum().item()

                if union == 0:
                    score = 1.0
                else:
                    score = (2.0 * inter) / (union + 1e-6)

                per_sample_dice.append(score)

            current_idx += batch_size

    # 3. Calculate Final Global Dice
    epsilon = 1e-6
    if union_sum == 0:
        global_dice = 1.0
    else:
        global_dice = (2.0 * intersection_sum) / (union_sum + epsilon)

    print(f"Final Validation Metric: {global_dice}")

    # 4. Correlation Analysis
    # Add dice scores to metadata dataframe
    # Note: val_loader shuffle=False ensures order matches
    if len(per_sample_dice) != len(meta_df):
        print(
            "Warning: Mismatch in validation samples and metadata rows. Skipping correlation analysis."
        )
        return global_dice

    meta_df["dice"] = per_sample_dice
    meta_df["error"] = 1.0 - meta_df["dice"]

    # Extract features
    # Timestamp to Hour of Day
    if "timestamp" in meta_df.columns:
        meta_df["datetime"] = pd.to_datetime(meta_df["timestamp"], unit="s")
        meta_df["hour_of_day"] = meta_df["datetime"].dt.hour

    # Features to correlate: hour_of_day, row_min (lat proxy), col_min (lon proxy)
    features = ["hour_of_day", "row_min", "col_min"]
    correlations = {}

    print("\nCorrelation between Model Error (1-Dice) and Metadata Features:")
    print(f"{'Feature':<20} | {'Pearson Corr':<12}")
    print("-" * 35)

    for feat in features:
        if feat in meta_df.columns:
            # Drop NaNs just in case
            valid_rows = meta_df[[feat, "error"]].dropna()
            if len(valid_rows) > 1:
                corr = valid_rows[feat].corr(valid_rows["error"])
                correlations[feat] = corr
                print(f"{feat:<20} | {corr:.4f}")
            else:
                print(f"{feat:<20} | N/A (Insufficient Data)")
        else:
            print(f"{feat:<20} | Not Found")

    return global_dice


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Train
    # We use Config.EPOCHS (20) to ensure convergence (Cite solution_lesson_node_00002)
    # Increased patience to handle metric volatility (Cite solution_lesson_node_00006)
    print("Starting Training Pipeline...")
    run_training(
        epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, early_stopping_patience=6
    )

    # 3. Load Best Model
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"\nLoading best model from {checkpoint_path} for analysis...")
    model = ResNet34UNet(
        in_channels=Config.IN_CHANNELS, out_channels=Config.CLASSES, pretrained=False
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    # 4. Validation & Failure Analysis
    val_metric = analyze_failures(model, device)

    # 5. Submission Logic
    # Threshold from task description
    THRESHOLD = 0.5973177358563411

    if val_metric > THRESHOLD:
        print(
            f"\nValidation Metric ({val_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(
            checkpoint_path=checkpoint_path,
            batch_size=Config.BATCH_SIZE,
            device=Config.DEVICE,
        )
    else:
        print(
            f"\nValidation Metric ({val_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
