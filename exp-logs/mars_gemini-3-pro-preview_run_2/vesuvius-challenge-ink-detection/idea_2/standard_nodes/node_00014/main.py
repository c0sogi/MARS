import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.train import train_model
from library.inference import predict_and_submit
from library.model import InkDetector
from library.dataset import InkDataset


def main():
    # --- 1. Configuration & Setup ---
    # Limit epochs for a fast baseline execution as requested
    Config.EPOCHS = 10

    Config.setup()
    set_seed(Config.SEED)

    print("--- Starting Pipeline ---")

    # --- 2. Training ---
    print("Step 1: Training Model...")
    train_model(load_cached_data=True)

    # --- 3. Validation & Assessment ---
    print("Step 2: Validating Best Model...")

    # Load Metadata
    val_csv_path = os.path.join(Config.METADATA_DIR, "validation.csv")
    if not os.path.exists(val_csv_path):
        raise FileNotFoundError("Validation metadata not found.")

    val_df = pd.read_csv(val_csv_path)

    # Initialize Dataset and Loader
    val_dataset = InkDataset(val_df, mode="validation", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model = InkDetector()
    model.to(Config.DEVICE)

    if os.path.exists(Config.BEST_MODEL_PATH):
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
    else:
        print("Warning: Best model not found. Using random weights.")

    model.eval()

    # Metrics Accumulators
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Failure Analysis Accumulators
    batch_errors = []
    batch_intensities = []

    with torch.no_grad():
        for images, labels, masks in val_loader:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # --- Metric Calculation (Global F0.5) ---
            preds_bin = (probs > Config.THRESHOLD).float()
            targets_bin = labels.float()

            tp = (preds_bin * targets_bin).sum().item()
            fp = (preds_bin * (1 - targets_bin)).sum().item()
            fn = ((1 - preds_bin) * targets_bin).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # --- Failure Analysis Data Collection ---
            # Error Magnitude: Mean Absolute Error per batch
            mae = torch.abs(probs - targets_bin).mean().item()
            batch_errors.append(mae)

            # Input Feature: Mean Intensity of the 3-channel input per batch
            mean_intensity = images.mean().item()
            batch_intensities.append(mean_intensity)

    # Compute Final F0.5 Score
    beta = 0.5
    beta_sq = beta**2
    smooth = 1e-6

    numerator = (1 + beta_sq) * total_tp
    denominator = (1 + beta_sq) * total_tp + beta_sq * total_fn + total_fp
    final_metric = (numerator + smooth) / (denominator + smooth)

    # Print Required Metric
    print(f"Final Validation Metric: {final_metric}")

    # --- 4. Failure Analysis ---
    print("Step 3: Failure Analysis...")
    if len(batch_errors) > 1:
        corr, p_val = pearsonr(batch_intensities, batch_errors)
        print(
            f"Correlation between Input Intensity and Model Error: {corr:.4f} (p={p_val:.4f})"
        )
        if abs(corr) > 0.3:
            print(
                "Observation: Significant correlation detected. Intensity normalization or augmentation may need adjustment."
            )
        else:
            print(
                "Observation: No strong linear correlation between intensity and error."
            )
    else:
        print("Insufficient batches for correlation analysis.")

    # --- 5. Submission ---
    BASELINE_THRESHOLD = 0.4738558828830719

    if final_metric > BASELINE_THRESHOLD:
        print(
            f"Validation score ({final_metric}) exceeds baseline ({BASELINE_THRESHOLD}). Generating submission..."
        )
        predict_and_submit(load_cached_data=True)
    else:
        print(
            f"Validation score ({final_metric}) did not exceed baseline ({BASELINE_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
