import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.train import train_model
from library.inference import run_inference
from library.dataset import InkDataset
from library.model import SegFormerMiTB2
from library.utils import set_seed


def main():
    # 1. Set Seed for Reproducibility
    set_seed(Config.SEED)

    # 2. Train Model
    # Using default epochs (15) as per Config, which is fast for the small dataset (412 patches).
    print("--- Starting Training ---")
    _ = train_model(debug=False, epochs=Config.EPOCHS)

    # 3. Validation and Failure Analysis
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load Validation Data
    # Using load_cached_data=True to utilize preprocessed .npy files if available
    df_val = pd.read_csv(Config.METADATA_VAL)
    val_dataset = InkDataset(df_val, split="val", load_cached_data=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    device = torch.device(Config.DEVICE)
    model = SegFormerMiTB2()
    model.to(device)

    weights_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(
            "Warning: Best model weights not found. Using random initialization for validation."
        )

    model.eval()

    # Metrics Accumulators
    tp_total = 0
    fp_total = 0
    fn_total = 0

    # Failure Analysis Accumulators
    errors = []
    features = []

    # Constants for F0.5 Score
    beta = 0.5
    threshold = 0.5
    smooth = 1e-6

    with torch.no_grad():
        for images, labels, masks in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # --- Metric Calculation (Global Micro-Average) ---
            preds_bin = (probs > threshold).float()

            # Flatten tensors
            p_flat = preds_bin.view(-1)
            t_flat = labels.view(-1)

            # Accumulate TP, FP, FN
            tp = (p_flat * t_flat).sum().item()
            fp = (p_flat * (1 - t_flat)).sum().item()
            fn = ((1 - p_flat) * t_flat).sum().item()

            tp_total += tp
            fp_total += fp
            fn_total += fn

            # --- Failure Analysis Data Collection ---
            # Error Magnitude: Mean Absolute Error per sample
            # Shape: (Batch, 1, H, W) -> Mean over spatial dims -> (Batch,)
            batch_mae = torch.abs(probs - labels).mean(dim=(1, 2, 3)).cpu().numpy()

            # Input Feature: Mean Intensity per sample
            # Shape: (Batch, 3, H, W) -> Mean over channel and spatial dims -> (Batch,)
            batch_intensity = images.mean(dim=(1, 2, 3)).cpu().numpy()

            errors.extend(batch_mae)
            features.extend(batch_intensity)

    # Calculate Final F0.5 Score
    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp_total
    denominator = (1 + beta_sq) * tp_total + (beta_sq * fn_total) + fp_total
    final_metric = (numerator + smooth) / (denominator + smooth)

    print(f"Final Validation Metric: {final_metric}")

    # Calculate Correlation for Failure Analysis
    if len(errors) > 1:
        # Pearson correlation coefficient
        corr_matrix = np.corrcoef(errors, features)
        correlation = corr_matrix[0, 1]
        print(
            f"Failure Analysis: Correlation between Input Intensity and Error Magnitude: {correlation:.10f}"
        )
    else:
        print("Failure Analysis: Insufficient data for correlation.")

    # 4. Submission Generation
    # Threshold defined in requirements
    SUBMISSION_THRESHOLD = 0.597622633

    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        # run_inference automatically loads the best model from cache
        run_inference()
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
