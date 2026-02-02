import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# Import from provided library
from library.config import Config
from library.dataset import ContrailDataset, get_valid_transform
from library.model import DeformableResNetUNet
from library.train_engine import run_training
from library.inference import run_inference
from library.utils import seed_everything


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Modify Config for Fast Baseline execution
    # We use a larger subset than the default debug size to ensure we can hit the metric
    # but keep epochs low to finish quickly.
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 4000  # Train on 4000 samples
    Config.EPOCHS = 8  # Train for 8 epochs
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 4

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"Starting Fast Baseline Run...")
    print(f"Config: {Config.DEBUG_SAMPLE_SIZE} samples, {Config.EPOCHS} epochs.")

    # ==========================================
    # 2. Training
    # ==========================================
    # run_training handles the training loop and saves 'best_model.pth'
    run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        debug=Config.DEBUG,
    )

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Validation and Failure Analysis...")

    device = torch.device(Config.DEVICE)

    # Load Validation Metadata
    valid_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)

    # Initialize Dataset with return_record_id=True to map back to metadata
    valid_dataset = ContrailDataset(
        valid_df,
        transform=get_valid_transform(),
        debug=False,  # We validate on the FULL validation set as required
        return_record_id=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model = DeformableResNetUNet(
        n_channels=Config.N_CHANNELS, n_classes=1, pretrained=False
    )
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Error: Best model not found. Using random weights.")

    model.to(device)
    model.eval()

    # Accumulators for Global Dice
    total_intersection = 0.0
    total_union = 0.0
    epsilon = 1e-6

    # Accumulators for Failure Analysis
    analysis_data = []

    # Pre-process metadata for lookup
    # Add datetime features to valid_df
    valid_df["datetime"] = pd.to_datetime(valid_df["timestamp"], unit="s")
    valid_df["hour"] = valid_df["datetime"].dt.hour
    valid_df_indexed = valid_df.set_index("record_id")

    with torch.no_grad():
        for images, masks, record_ids in valid_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Inference (No TTA for validation to save time, or match training validation)
            with autocast():
                logits = model(images)
                probs = torch.sigmoid(logits)

            preds = (probs > Config.THRESHOLD).float()

            # --- Global Dice Calculation ---
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            batch_intersection = (preds_flat * masks_flat).sum().item()
            batch_union = preds_flat.sum().item() + masks_flat.sum().item()

            total_intersection += batch_intersection
            total_union += batch_union

            # --- Failure Analysis Data Collection ---
            # We process per-sample to correlate errors with metadata
            # Move to CPU for analysis
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i, rid in enumerate(record_ids):
                p = preds_np[i, 0].flatten()
                m = masks_np[i, 0].flatten()

                intersection = np.sum(p * m)
                union = np.sum(p) + np.sum(m)
                dice = (2.0 * intersection + epsilon) / (union + epsilon)
                error = 1.0 - dice

                # Ground truth fraction (how much contrail is there?)
                gt_fraction = np.mean(m)

                # Get metadata
                try:
                    meta_row = valid_df_indexed.loc[int(rid)]
                    hour = meta_row["hour"]
                    timestamp = meta_row["timestamp"]
                except:
                    hour = -1
                    timestamp = -1

                analysis_data.append(
                    {
                        "record_id": rid,
                        "dice": dice,
                        "error": error,
                        "gt_fraction": gt_fraction,
                        "hour": hour,
                        "timestamp": timestamp,
                    }
                )

    # Compute Final Metric
    final_metric = (2.0 * total_intersection + epsilon) / (total_union + epsilon)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis Report
    # ==========================================
    print("\n--- Failure Analysis ---")
    analysis_df = pd.DataFrame(analysis_data)

    if not analysis_df.empty:
        # Correlation with Error
        # We check: Hour, Timestamp, GT Fraction
        # Note: GT Fraction correlation often highlights if model struggles with small vs large objects
        correlations = analysis_df[
            ["error", "hour", "timestamp", "gt_fraction"]
        ].corr()["error"]

        print("Correlation between Model Error (1-Dice) and Features:")
        print(correlations.drop("error").sort_values(ascending=False))

        # Additional Insight: Error by Hour
        # print("\nMean Error by Hour:")
        # print(analysis_df.groupby('hour')['error'].mean())
    else:
        print("No analysis data collected.")

    # ==========================================
    # 5. Submission
    # ==========================================
    THRESHOLD_SCORE = 0.5676456935477064

    if final_metric > THRESHOLD_SCORE:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        # Run inference on Test Set
        # We use the library function which handles TTA and RLE encoding
        run_inference(
            checkpoint_path=Config.BEST_MODEL_PATH,
            output_path=Config.SUBMISSION_PATH,
            batch_size=Config.BATCH_SIZE,
            device=Config.DEVICE,
            threshold=Config.THRESHOLD,
            debug=False,  # Must run on full test set
        )
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD_SCORE}). Skipping submission."
        )


if __name__ == "__main__":
    main()
