import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import set_seed
from library.train import run_training
from library.inference import generate_submission
from library.model import ContextEnhancedUNet
from library.dataset import ContrailDataset, get_transforms


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # Override Config for Fast Baseline within 2 hours
    # 30 epochs on ~18k images with ResNet18 is estimated to take ~45 mins on A100
    Config.EPOCHS = 30
    Config.NUM_WORKERS = 4  # Utilize more vCPUs

    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # ==========================================
    # 2. Training Pipeline
    # ==========================================
    print("\nStarting Training Pipeline...")
    # run_training handles the training loop and saves the best model to Config.BEST_MODEL_PATH
    run_training(debug=False)

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Validation and Failure Analysis...")
    device = torch.device(Config.DEVICE)

    # Load Best Model
    model = ContextEnhancedUNet(in_channels=Config.IN_CHANNELS, pretrained=False)
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Error: Best model not found at {Config.BEST_MODEL_PATH}")
        return

    print(f"Loading weights from {Config.BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Initialize Validation Loader
    val_dataset = ContrailDataset(
        split="validation", transform=get_transforms("validation", Config), debug=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Metrics Accumulators
    total_intersection = 0.0
    total_union = 0.0
    smooth = 1e-6

    # Failure Analysis Data
    sample_errors = []
    record_ids_list = []

    # Inference Loop (No Grad)
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            r_ids = batch["record_id"]

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > Config.THRESHOLD).float()

            # 1. Update Global Dice Stats
            preds_flat = preds.view(-1)
            targets_flat = masks.view(-1)

            intersection = (preds_flat * targets_flat).sum().item()
            union = preds_flat.sum().item() + targets_flat.sum().item()

            total_intersection += intersection
            total_union += union

            # 2. Per-Sample Analysis
            # Calculate Dice per sample to determine error magnitude
            # Shape: (B, C, H, W) -> flatten spatial dims -> (B, -1)
            B = images.size(0)
            p_flat = preds.view(B, -1)
            t_flat = masks.view(B, -1)

            i_s = (p_flat * t_flat).sum(dim=1)
            u_s = p_flat.sum(dim=1) + t_flat.sum(dim=1)
            dice_s = (2.0 * i_s + smooth) / (u_s + smooth)

            # Error metric: 1.0 - Dice Score
            # High error means low Dice (poor overlap)
            errors = 1.0 - dice_s.cpu().numpy()

            sample_errors.extend(errors)
            record_ids_list.extend(r_ids)

    # Compute Final Global Metric
    final_metric = (2.0 * total_intersection + smooth) / (total_union + smooth)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis Correlations ---
    print("\nPerforming Failure Analysis...")

    # Load metadata
    if not os.path.exists(Config.VAL_METADATA_PATH):
        print("Warning: Validation metadata not found. Skipping correlation analysis.")
    else:
        val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)

        # Create DataFrame of errors
        error_df = pd.DataFrame(
            {"record_id": [str(x) for x in record_ids_list], "error": sample_errors}
        )

        # Ensure record_id is string for merging
        val_meta_df["record_id"] = val_meta_df["record_id"].astype(str)

        # Merge metadata with errors
        analysis_df = val_meta_df.merge(error_df, on="record_id", how="inner")

        # Feature Engineering for correlation
        if "timestamp" in analysis_df.columns:
            analysis_df["datetime"] = pd.to_datetime(analysis_df["timestamp"], unit="s")
            analysis_df["hour"] = analysis_df["datetime"].dt.hour
            analysis_df["month"] = analysis_df["datetime"].dt.month

        # Calculate correlations
        features_to_check = ["timestamp", "row_min", "col_min", "hour", "month"]
        print("Correlation between Model Error (1-Dice) and Metadata Features:")

        for feat in features_to_check:
            if feat in analysis_df.columns:
                # Drop NaNs just in case
                valid_data = analysis_df[[feat, "error"]].dropna()
                if len(valid_data) > 1:
                    corr = valid_data[feat].corr(valid_data["error"])
                    print(f"  {feat}: {corr:.6f}")
                else:
                    print(f"  {feat}: Not enough data")
            else:
                print(f"  {feat}: Feature not found")

    # ==========================================
    # 4. Submission
    # ==========================================
    THRESHOLD_SCORE = 0.5676456935477064

    if final_metric > THRESHOLD_SCORE:
        print(
            f"\nValidation Metric ({final_metric}) exceeds threshold ({THRESHOLD_SCORE})."
        )
        print("Generating submission for test set...")
        generate_submission(debug=False)
    else:
        print(
            f"\nValidation Metric ({final_metric}) does not exceed threshold ({THRESHOLD_SCORE})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
