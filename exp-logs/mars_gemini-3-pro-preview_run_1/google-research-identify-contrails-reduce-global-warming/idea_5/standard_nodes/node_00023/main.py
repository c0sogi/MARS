import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import ContrailDataset
from library.model import TemporalAshNet
from library.train import run_training
from library.inference import run_inference

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # --------------------------------------------------------------------------
    Config.setup()

    # Training on full dataset with configuration defined in library/config.py
    print(f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Training Phase
    # --------------------------------------------------------------------------
    print("\n=== Starting Training ===")
    # Using full dataset and extended patience for metric-coupled scheduling
    run_training(debug=False, early_stopping_patience=10)

    # --------------------------------------------------------------------------
    # 3. Validation & Failure Analysis Phase
    # --------------------------------------------------------------------------
    print("\n=== Starting Validation & Failure Analysis ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found. Training may have failed.")
        return

    # Load Model
    model = TemporalAshNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load Full Validation Dataset
    val_dataset = ContrailDataset(Config.VAL_METADATA_PATH, stage="validation")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Variables for Global Dice
    global_intersection = 0.0
    global_union = 0.0

    # Variables for Failure Analysis
    sample_records = []
    sample_errors = []

    print(f"Evaluating on {len(val_dataset)} validation samples...")

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            record_ids = batch["record_id"]

            # Inference
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > Config.THRESHOLD).float()

            # --- Global Metric Accumulation ---
            preds_flat = preds.view(-1)
            targets_flat = masks.view(-1)

            intersection = (preds_flat * targets_flat).sum().item()
            union = preds_flat.sum().item() + targets_flat.sum().item()

            global_intersection += intersection
            global_union += union

            # --- Per-Sample Failure Analysis ---
            # Calculate Dice per image to correlate with metadata
            B = images.size(0)
            p_flat = preds.view(B, -1)
            t_flat = masks.view(B, -1)

            # Intersection and Union per sample
            i_s = (p_flat * t_flat).sum(dim=1).cpu().numpy()
            u_s = p_flat.sum(dim=1).cpu().numpy() + t_flat.sum(dim=1).cpu().numpy()

            # Dice = 2*I / (U + eps)
            # Error = 1 - Dice
            dices = (2.0 * i_s) / (u_s + 1e-6)
            errors = 1.0 - dices

            sample_records.extend([str(r) for r in record_ids])
            sample_errors.extend(errors)

    # Compute Final Global Dice
    epsilon = 1e-6
    final_metric = (2.0 * global_intersection) / (global_union + epsilon)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric:.18f}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis (Correlation with Error Magnitude) ---")

    # Create DataFrame of errors
    error_df = pd.DataFrame({"record_id": sample_records, "error": sample_errors})

    # Merge with validation metadata
    val_meta_df = val_dataset.df.copy()
    val_meta_df["record_id"] = val_meta_df["record_id"].astype(str)

    analysis_df = val_meta_df.merge(error_df, on="record_id", how="inner")

    # Features to analyze
    features = ["timestamp", "row_min", "col_min", "row_size", "col_size"]

    for feat in features:
        if feat in analysis_df.columns:
            # Drop NaNs for correlation calculation
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr = valid_data[feat].corr(valid_data["error"])
                print(f"Feature '{feat}': Correlation = {corr:.4f}")
            else:
                print(f"Feature '{feat}': Insufficient data")
        else:
            print(f"Feature '{feat}': Not found in metadata")

    # --------------------------------------------------------------------------
    # 4. Submission Phase
    # --------------------------------------------------------------------------
    THRESHOLD_SCORE = 0.5973177358563411

    if final_metric > THRESHOLD_SCORE:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({THRESHOLD_SCORE:.6f}). Generating Submission..."
        )
        # Run inference on test set
        # load_cached_data=False ensures we use the fresh model predictions
        run_inference(
            checkpoint_path=checkpoint_path,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            load_cached_data=False,
        )
    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({THRESHOLD_SCORE:.6f}). Submission Skipped."
        )


if __name__ == "__main__":
    main()
