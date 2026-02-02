import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import set_seed, get_logger
from library.dataset import ContrailDataset, get_transforms
from library.model import DualStreamUNet
from library.train import train_model
from library.predict import predict_and_submit


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Adjust configuration for a fast baseline run within time limits
    # We reduce epochs to 8 to ensure completion within ~2 hours while maintaining performance.
    Config.EPOCHS = 8

    # We use the full dataset (debug=False) to ensure we meet the performance threshold.
    # The dataset size (18k) is manageable for 8 epochs on A100.
    DEBUG_MODE = False

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # ==========================================
    # 2. Training
    # ==========================================
    print(f"Starting training for {Config.EPOCHS} epochs...")
    # train_model handles the training loop, validation monitoring, and saving the best model.
    train_model(debug=DEBUG_MODE)

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("Starting validation inference and failure analysis...")

    device = torch.device(Config.DEVICE)

    # Load the best model saved during training
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at {Config.BEST_MODEL_PATH}")

    model = SingleStreamUNet(
        backbone_name=Config.BACKBONE,
        pretrained=False,  # Weights are loaded from checkpoint
        in_chans=Config.IN_CHANNELS,
    )
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Setup Validation Loader
    val_dataset = ContrailDataset(
        metadata_path=Config.VALID_METADATA_PATH,
        split="validation",
        transform=get_transforms("validation"),
        debug=DEBUG_MODE,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Inference Loop
    # We need to compute Global Dice and per-sample error
    intersection_sum = 0.0
    union_sum = 0.0

    sample_errors = []

    with torch.no_grad():
        for images, masks, record_ids in val_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > Config.THRESHOLD).float()

            # --- Global Metrics Accumulation ---
            preds_flat = preds.view(-1)
            targets_flat = masks.view(-1)

            intersection_sum += (preds_flat * targets_flat).sum().item()
            union_sum += preds_flat.sum().item() + targets_flat.sum().item()

            # --- Per-Sample Analysis ---
            # Iterate over the batch to calculate individual errors
            batch_size = images.size(0)
            for i in range(batch_size):
                p = preds[i].view(-1)
                t = masks[i].view(-1)

                inter = (p * t).sum().item()
                union = p.sum().item() + t.sum().item()

                # Dice score for this specific sample
                # Handle empty union case (both empty) -> score 1.0
                if union == 0:
                    dice = 1.0
                else:
                    dice = (2.0 * inter) / union

                # Error is 1 - Dice
                error = 1.0 - dice

                sample_errors.append({"record_id": str(record_ids[i]), "error": error})

    # Compute Final Global Dice
    if union_sum == 0:
        final_metric = 1.0
    else:
        final_metric = (2.0 * intersection_sum) / union_sum

    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Load metadata to correlate errors with features
    df_errors = pd.DataFrame(sample_errors)
    df_meta = pd.read_csv(Config.VALID_METADATA_PATH)
    df_meta["record_id"] = df_meta["record_id"].astype(str)

    # Merge errors with metadata
    df_analysis = df_errors.merge(df_meta, on="record_id", how="left")

    print("Failure Analysis - Correlation of Error with Metadata Features:")
    features_to_check = ["timestamp", "row_min", "col_min", "row_size", "col_size"]

    for feat in features_to_check:
        if feat in df_analysis.columns:
            # Drop NaNs just in case
            valid_data = df_analysis[[feat, "error"]].dropna()
            if not valid_data.empty:
                corr = valid_data[feat].corr(valid_data["error"])
                print(f"  Correlation with {feat}: {corr:.6f}")
            else:
                print(f"  Correlation with {feat}: N/A (No valid data)")

    # ==========================================
    # 4. Submission
    # ==========================================
    THRESHOLD_SCORE = 0.5910660985501295

    if final_metric > THRESHOLD_SCORE:
        print(
            f"Validation metric ({final_metric:.6f}) exceeds threshold ({THRESHOLD_SCORE}). Generating submission..."
        )
        predict_and_submit(load_cached_data=True)
    else:
        print(
            f"Validation metric ({final_metric:.6f}) does not exceed threshold ({THRESHOLD_SCORE}). Skipping submission."
        )


if __name__ == "__main__":
    main()
