import os
import pandas as pd
import numpy as np
import torch
import scipy.stats as stats

from library.config import Config
from library.train import run_training, generate_submission
from library.model import ResidualRefinedNet
from library.data import load_or_process_data, get_dataloaders
from library.utils import GlobalMCRMSE


def main():
    # ==============================================================================
    # 1. Training
    # ==============================================================================
    # We use the default configuration (20 epochs) which is optimized for this dataset size.
    # This ensures a fast baseline execution while allowing convergence.
    print("Starting training pipeline...")
    run_training(epochs=Config.EPOCHS)

    # ==============================================================================
    # 2. Validation & Metric Calculation
    # ==============================================================================
    print("Evaluating best model on validation set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model saved during training
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = ResidualRefinedNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get validation data
    # We use the loader for batching and the raw data function to get IDs for analysis
    _, val_loader = get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True)
    _, _, _, _, val_ids = load_or_process_data(mode="val", load_cached_data=True)

    # Initialize metric calculator
    # Scored columns indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    metric_calc = GlobalMCRMSE(scored_indices=[0, 1, 3], device=device)

    # List to store per-sample error for failure analysis
    sample_errors = []

    with torch.no_grad():
        for features, pair_indices, targets, mask in val_loader:
            features = features.to(device)
            pair_indices = pair_indices.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            # Inference
            preds = model(features, pair_indices)

            # Update Global Metric
            metric_calc.update(preds, targets, mask)

            # --- Calculate Per-Sample Error for Failure Analysis ---
            # We calculate the mean RMSE across the scored columns for each sample in the batch
            # Shape: (B, L, 5)
            mse = (preds - targets) ** 2
            scored_cols = [0, 1, 3]

            for i in range(preds.shape[0]):
                sample_rmse_sum = 0.0
                valid_col_count = 0

                for col in scored_cols:
                    m_col = mask[i, :, col]
                    if m_col.sum() > 0:
                        col_rmse = torch.sqrt(
                            (mse[i, :, col] * m_col).sum() / m_col.sum()
                        )
                        sample_rmse_sum += col_rmse
                        valid_col_count += 1

                if valid_col_count > 0:
                    sample_errors.append((sample_rmse_sum / valid_col_count).item())
                else:
                    sample_errors.append(0.0)

    # Compute and print final metric
    final_metric = metric_calc.compute()
    print(f"Final Validation Metric: {final_metric}")

    # ==============================================================================
    # 3. Failure Analysis
    # ==============================================================================
    print("Performing failure analysis...")

    # Load validation metadata to get signal_to_noise
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Create a DataFrame linking IDs to their computed errors
    # Note: val_loader (shuffle=False) order matches val_ids order
    error_df = pd.DataFrame({"id": val_ids, "model_error": sample_errors})

    # Merge with metadata
    analysis_df = pd.merge(error_df, val_df, on="id", how="inner")

    # Calculate correlation
    if "signal_to_noise" in analysis_df.columns:
        corr, _ = stats.pearsonr(
            analysis_df["model_error"], analysis_df["signal_to_noise"]
        )
        print(f"Correlation between Error and Signal_to_Noise: {corr}")
    else:
        print("signal_to_noise column not found for correlation analysis.")

    # ==============================================================================
    # 4. Submission
    # ==============================================================================
    THRESHOLD = 0.5421870350837708

    if final_metric < THRESHOLD:
        print(f"Metric passed threshold ({THRESHOLD}). Generating submission...")
        generate_submission()
    else:
        print(
            f"Metric {final_metric} failed to meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
