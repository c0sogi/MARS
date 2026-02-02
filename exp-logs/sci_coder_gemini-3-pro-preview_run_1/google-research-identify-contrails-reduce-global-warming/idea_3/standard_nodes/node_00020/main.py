import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import library modules
from library.config import Config
from library.utils import set_seed, rle_encode
from library.train import train_model
from library.model import DilatedResNetUNet
from library.dataset import get_dataloaders


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    # Set random seeds for reproducibility
    set_seed(Config.SEED)

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    # We run for a limited number of epochs (3) to create a fast baseline.
    # The A100 GPU can handle the full dataset efficiently.
    print("\n==== Starting Training ====")
    best_dice_score = train_model(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        debug=False,  # Use full dataset for a robust baseline
    )

    # --------------------------------------------------------------------------
    # 3. Load Best Model
    # --------------------------------------------------------------------------
    print("\n==== Loading Best Model ====")
    model = DilatedResNetUNet().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint file not found.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # --------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n==== Running Validation & Failure Analysis ====")

    # Get validation loader (shuffle=False preserves order)
    _, val_loader, _ = get_dataloaders(batch_size=Config.BATCH_SIZE, debug=False)

    # Metadata for mapping predictions back to record IDs and features
    val_meta = val_loader.dataset.metadata.copy()
    # Ensure record_id is string for merging
    val_meta["record_id"] = val_meta["record_id"].astype(str)

    intersection_sum = 0.0
    union_sum = 0.0

    sample_stats = []
    current_idx = 0

    # Disable gradients for inference
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Inference
            logits = model(images)
            preds = torch.sigmoid(logits)
            preds_bin = (preds > Config.THRESHOLD).float()

            # --- Global Metric Calculation ---
            preds_flat = preds_bin.view(-1)
            masks_flat = masks.view(-1)

            intersection_sum += (preds_flat * masks_flat).sum().item()
            union_sum += preds_flat.sum().item() + masks_flat.sum().item()

            # --- Per-Sample Analysis ---
            batch_size = images.size(0)
            for b in range(batch_size):
                # Flatten single sample
                p = preds_bin[b].view(-1)
                m = masks[b].view(-1)

                inter = (p * m).sum().item()
                union = p.sum().item() + m.sum().item()

                # Dice per sample
                dice = (2.0 * inter + 1e-6) / (union + 1e-6)
                error = 1.0 - dice

                # Retrieve record_id using linear index
                record_id = val_meta.iloc[current_idx + b]["record_id"]

                sample_stats.append(
                    {"record_id": str(record_id), "sample_dice": dice, "error": error}
                )

            current_idx += batch_size

    # Compute Final Global Dice
    smooth = 1e-6
    final_metric = (2.0 * intersection_sum + smooth) / (union_sum + smooth)

    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n==== Failure Analysis ====")
    stats_df = pd.DataFrame(sample_stats)

    # Merge error stats with metadata
    analysis_df = pd.merge(stats_df, val_meta, on="record_id", how="left")

    # Add temporal features if timestamp exists
    if "timestamp" in analysis_df.columns:
        analysis_df["datetime"] = pd.to_datetime(analysis_df["timestamp"], unit="s")
        analysis_df["hour_of_day"] = analysis_df["datetime"].dt.hour

    # Calculate correlations with error magnitude
    # We exclude non-numeric columns and the target metrics themselves
    exclude_cols = [
        "record_id",
        "sample_dice",
        "error",
        "projection_wkt",
        "datetime",
        "human_pixel_masks",
        "human_individual_masks",
    ]
    # Also exclude band paths
    exclude_cols += [c for c in analysis_df.columns if "band_" in c]

    numeric_cols = [
        c
        for c in analysis_df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(analysis_df[c])
    ]

    print("Correlation between Error (1 - Dice) and Metadata Features:")
    correlations = {}
    for col in numeric_cols:
        if analysis_df[col].nunique() > 1:
            corr = analysis_df["error"].corr(analysis_df[col])
            correlations[col] = corr
            print(f"  {col:<20}: {corr:.4f}")
        else:
            print(f"  {col:<20}: N/A (Constant)")

    # --------------------------------------------------------------------------
    # 5. Submission Generation
    # --------------------------------------------------------------------------
    target_threshold = 0.5973177358563411

    if final_metric > target_threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({target_threshold}). Generating submission..."
        )

        # Get Test Loader
        _, _, test_loader = get_dataloaders(batch_size=Config.BATCH_SIZE, debug=False)

        submission_data = []

        with torch.no_grad():
            for images, record_ids in tqdm(test_loader, desc="Inference (Test)"):
                images = images.to(device, dtype=torch.float32)

                logits = model(images)
                preds = torch.sigmoid(logits)

                # Convert to binary mask on CPU
                preds_bin = (preds > Config.THRESHOLD).cpu().numpy()

                for i, record_id in enumerate(record_ids):
                    # Extract mask for single image (H, W)
                    mask = preds_bin[i, 0]

                    # Encode
                    encoded = rle_encode(mask)
                    submission_data.append(
                        {"record_id": record_id, "encoded_pixels": encoded}
                    )

        # Save Submission
        sub_df = pd.DataFrame(submission_data)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({target_threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
