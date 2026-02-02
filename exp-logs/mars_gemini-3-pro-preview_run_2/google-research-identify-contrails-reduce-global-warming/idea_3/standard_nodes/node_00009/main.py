import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import from library
from library.config import Config
from library.utils import set_seed, dice_coef_metric
from library.dataset import ContrailDataset
from library.model import MultiTaskResNetUNet
from library.train import train_model, inference


def main():
    # 1. Setup and Config Overrides
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Override Config for Fast Baseline Execution within time limits
    # Strategy dictates full dataset, but we limit epochs to ensure < 2h runtime.
    Config.EPOCHS = 10

    print("Configuration:")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Device: {Config.DEVICE}")

    # 2. Train the Model
    print("\nStarting Training...")
    # train_model handles the training loop and returns path to best checkpoint
    # We use debug=False to use the full dataset as per the Strategy
    best_model_path = train_model(debug=False)

    # 3. Validation & Failure Analysis
    print("\nStarting Validation and Failure Analysis...")

    device = torch.device(Config.DEVICE)

    # Load the best model
    model = MultiTaskResNetUNet(in_channels=Config.IN_CHANNELS, pretrained=False)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Setup Validation Data
    val_dataset = ContrailDataset(split="validation")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Accumulators for Global Dice
    intersection_sum = 0.0
    union_sum = 0.0

    # Accumulators for Failure Analysis
    # We will store record_id and error (1 - dice) per sample
    sample_errors = []
    sample_ids = []

    with torch.no_grad():
        for images, masks, labels, record_ids in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            seg_logits, cls_logits = model(images)

            # Apply Gated Inference Logic
            seg_probs = torch.sigmoid(seg_logits)
            cls_probs = torch.sigmoid(cls_logits)

            # Pixel Threshold
            pred_masks = (seg_probs > Config.PIXEL_THRESHOLD).float()

            # Classification Gate
            # If image-level prob < threshold, zero out
            gate_mask = (cls_probs > Config.CLS_THRESHOLD).float().view(-1, 1, 1, 1)
            pred_masks = pred_masks * gate_mask

            # --- Global Dice Calculation ---
            pred_flat = pred_masks.view(-1)
            true_flat = masks.view(-1)

            intersection_sum += (pred_flat * true_flat).sum().item()
            union_sum += pred_flat.sum().item() + true_flat.sum().item()

            # --- Per-Sample Analysis ---
            # Calculate Dice per image for error correlation
            # Flatten per sample: (B, -1)
            p_flat = pred_masks.view(images.size(0), -1)
            t_flat = masks.view(images.size(0), -1)

            inter_per_sample = (p_flat * t_flat).sum(dim=1)
            union_per_sample = p_flat.sum(dim=1) + t_flat.sum(dim=1)

            smooth = 1e-6
            dice_per_sample = (2.0 * inter_per_sample + smooth) / (
                union_per_sample + smooth
            )

            # Error = 1 - Dice
            errors = 1.0 - dice_per_sample.cpu().numpy()

            sample_errors.extend(errors)
            sample_ids.extend(record_ids)

    # Compute Final Global Dice
    smooth = 1e-6
    final_metric = (2.0 * intersection_sum + smooth) / (union_sum + smooth)

    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame({"record_id": sample_ids, "error": sample_errors})

    # Merge with metadata to get features
    # val_dataset.df contains the metadata
    df_meta = val_dataset.df.copy()
    df_merged = df_analysis.merge(df_meta, on="record_id", how="left")

    # Feature Engineering for Correlation
    if "timestamp" in df_merged.columns:
        df_merged["datetime"] = pd.to_datetime(df_merged["timestamp"], unit="s")
        df_merged["hour"] = df_merged["datetime"].dt.hour
    else:
        # Fallback if timestamp missing (unlikely given metadata)
        df_merged["hour"] = 0

    # Calculate Correlations
    # We look for correlation between Error and Metadata features
    features = ["hour", "row_min", "col_min"]
    correlations = {}

    for feat in features:
        if feat in df_merged.columns:
            # Drop NaNs just in case
            valid_df = df_merged[[feat, "error"]].dropna()
            if not valid_df.empty:
                corr = valid_df[feat].corr(valid_df["error"])
                correlations[feat] = corr
            else:
                correlations[feat] = 0.0
        else:
            correlations[feat] = 0.0

    print("Correlation between Error Magnitude (1-Dice) and Input Features:")
    for feat, corr in correlations.items():
        print(f"  {feat}: {corr:.4f}")

    # 4. Submission
    # Threshold check
    THRESHOLD = 0.5454606988733747

    if final_metric > THRESHOLD:
        print(
            f"\nValidation Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        inference(best_model_path)
    else:
        print(
            f"\nValidation Metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
