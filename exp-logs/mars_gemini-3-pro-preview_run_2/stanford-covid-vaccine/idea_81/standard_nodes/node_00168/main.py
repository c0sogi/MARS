import os
import sys
import numpy as np
import pandas as pd
import torch

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, compute_global_rmse
from library.data import get_dataloaders
from library.model import AHDRNModel
from library.engine import train_model, generate_submission


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for runtime constraints
    Config.NUM_EPOCHS = 20
    Config.PATIENCE = 5

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    print(f"Starting execution with Device: {Config.DEVICE}")
    print(f"Training for max {Config.NUM_EPOCHS} epochs...")

    # ==========================================
    # 2. Training Pipeline
    # ==========================================
    # train_model handles data loading, training loop, and saving best_model.pth
    best_metric = train_model()

    # REQUIRED: Print the final validation metric in the specific format
    print(f"Final Validation Metric: {best_metric}")

    # ==========================================
    # 3. Failure Analysis
    # ==========================================
    print("\n==== Starting Failure Analysis ====")

    device = torch.device(Config.DEVICE)

    # Load Validation Data
    # We use get_dataloaders to ensure consistent preprocessing
    _, val_loader, _ = get_dataloaders()

    # Load Best Model
    model = AHDRNModel().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model not found. Using untrained model for analysis.")

    model.eval()

    all_preds = []
    all_targets = []

    # Inference on Validation Set
    print("Running inference on validation set...")
    with torch.no_grad():
        for inputs, partner_indices, targets in val_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Forward pass (returns y_2 in eval mode)
            preds = model(inputs, partner_indices)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Per-Sample RMSE
    # Filter for scored columns and positions
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # Slice: (N, 68, Scored_Cols)
    preds_scored = all_preds[:, : Config.PRED_LEN, scored_indices]
    targets_scored = all_targets[:, : Config.PRED_LEN, scored_indices]

    # MSE per sample (average over positions and columns)
    mse_per_sample = np.mean((preds_scored - targets_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load Metadata for Correlation
    val_df = pd.read_csv(Config.VAL_CSV)

    # Check alignment
    if len(val_df) != len(rmse_per_sample):
        print(
            f"Error: Mismatch in validation set size. DF: {len(val_df)}, Preds: {len(rmse_per_sample)}"
        )
    else:
        val_df["rmse"] = rmse_per_sample

        # Feature Engineering for Analysis
        val_df["gc_content"] = val_df["sequence"].apply(
            lambda s: (s.count("G") + s.count("C")) / len(s)
        )

        # Calculate Correlations
        corr_sn = val_df["signal_to_noise"].corr(val_df["rmse"])
        corr_gc = val_df["gc_content"].corr(val_df["rmse"])

        print("Correlations between Error (RMSE) and Input Features:")
        print(f"  Signal-to-Noise Ratio: {corr_sn:.4f}")
        print(f"  GC Content:            {corr_gc:.4f}")

        # Insight
        if corr_sn < 0:
            print(
                "  -> Negative correlation with S/N implies higher signal quality leads to lower error."
            )

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    THRESHOLD = 0.47142532743789534

    if best_metric < THRESHOLD:
        print(
            f"\nMetric {best_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric {best_metric} does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
