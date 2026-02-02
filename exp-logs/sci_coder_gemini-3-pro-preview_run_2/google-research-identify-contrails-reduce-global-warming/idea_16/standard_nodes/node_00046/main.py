import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, dice_coef
from library.dataset import ContrailDataset
from library.model import ConvNeXtUNet
from library.train import train_model
from library.inference import inference


def perform_failure_analysis(model, val_loader, val_df, device):
    """
    Computes Global Dice and performs failure analysis.
    """
    model.eval()

    # Accumulators for Global Dice
    total_intersection = 0.0
    total_union = 0.0

    # Accumulators for Failure Analysis
    sample_errors = []
    sample_meta = []

    # Map record_id to metadata row for quick lookup
    # val_df has columns: record_id, timestamp, row_min, col_min, etc.
    # Ensure record_id is string for consistent mapping
    val_df["record_id"] = val_df["record_id"].astype(str)
    meta_map = val_df.set_index("record_id").to_dict("index")

    print("Running validation and failure analysis...")

    with torch.no_grad():
        for images, masks, record_ids in val_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            with autocast(enabled=True):
                logits = model(images)
                probs = torch.sigmoid(logits)
                preds = (probs > Config.THRESHOLD).float()

            # --- Global Dice Calculation ---
            preds_flat = preds.view(preds.size(0), -1)
            masks_flat = masks.view(masks.size(0), -1)

            intersection = (preds_flat * masks_flat).sum(dim=1)
            union = (preds_flat + masks_flat).sum(dim=1)

            total_intersection += intersection.sum().item()
            total_union += union.sum().item()

            # --- Failure Analysis Data Collection ---
            # Calculate per-sample Dice to define "Error" = 1 - Dice
            # Handle division by zero for per-sample dice
            sample_dice = (2.0 * intersection + Config.SMOOTH) / (union + Config.SMOOTH)
            sample_error = 1.0 - sample_dice.cpu().numpy()

            for i, rid in enumerate(record_ids):
                # Ensure rid is string
                rid_str = str(rid)
                if rid_str in meta_map:
                    row = meta_map[rid_str]

                    # Extract features
                    # Timestamp to Hour
                    ts = row.get("timestamp", 0)
                    hour = (ts % 86400) / 3600.0

                    # Spatial
                    r_min = row.get("row_min", 0)
                    c_min = row.get("col_min", 0)

                    sample_errors.append(sample_error[i])
                    sample_meta.append(
                        {
                            "hour": hour,
                            "row_min": r_min,
                            "col_min": c_min,
                            "contrail_size": masks_flat[i].sum().item(),
                        }
                    )

    # Compute Global Dice
    global_dice = (2.0 * total_intersection + Config.SMOOTH) / (
        total_union + Config.SMOOTH
    )

    # Compute Correlations
    if len(sample_errors) > 0:
        df_analysis = pd.DataFrame(sample_meta)
        df_analysis["error"] = sample_errors

        print("\nFailure Analysis (Correlation with Error):")
        # Check correlations with available columns
        for col in ["hour", "row_min", "col_min", "contrail_size"]:
            if col in df_analysis.columns and df_analysis[col].nunique() > 1:
                corr = df_analysis[col].corr(df_analysis["error"])
                print(f"  {col}: {corr:.4f}")
            else:
                print(f"  {col}: N/A (Constant or Missing)")

    return global_dice


def main():
    # 1. Setup and Config Overrides for Fast Baseline
    set_seed(Config.SEED)

    # Override Config for speed
    Config.EPOCHS = 5
    Config.MAX_TRAIN_SAMPLES = 5000
    # Keep Batch Size 32 as per Config

    print(f"Configuration Overrides:")
    print(f"  EPOCHS: {Config.EPOCHS}")
    print(f"  MAX_TRAIN_SAMPLES: {Config.MAX_TRAIN_SAMPLES}")

    # 2. Train the Model
    # This saves the best model to Config.BEST_MODEL_PATH
    train_model()

    # 3. Load Best Model for Validation
    device = torch.device(Config.DEVICE)
    model = ConvNeXtUNet()
    model.to(device)

    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}")
        checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print("Error: Best model not found after training.")
        return

    # 4. Prepare Validation Data
    val_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)
    # Use full validation set for final metric
    val_dataset = ContrailDataset(val_df, stage="validation", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Compute Metric and Failure Analysis
    final_metric = perform_failure_analysis(model, val_loader, val_df, device)

    # Print Exact Metric Requirement
    print(f"Final Validation Metric: {final_metric}")

    # 6. Conditional Submission
    THRESHOLD = 0.5910660985501295

    if final_metric > THRESHOLD:
        print(f"Metric {final_metric} > {THRESHOLD}. Generating submission...")
        inference()
    else:
        print(f"Metric {final_metric} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
