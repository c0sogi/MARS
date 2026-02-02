import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure local library can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, mcrmse_loss
from library.data import get_loaders, get_test_loader
from library.model import InterleavedBiGRU
from library.engine import train_model, inference


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Device configuration
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # Override Config for fast baseline execution as per requirements
    Config.EPOCHS = 10
    print(f"Training for {Config.EPOCHS} epochs.")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Loading datasets...")
    # Load cached data if available to save time
    train_loader, val_loader = get_loaders(load_cached_data=True)

    # =========================================================================
    # 3. Model Initialization & Training
    # =========================================================================
    print("Initializing InterleavedBiGRU model...")
    model = InterleavedBiGRU()

    print("Starting training loop...")
    train_model(model, train_loader, val_loader)

    # =========================================================================
    # 4. Validation & Metric Calculation
    # =========================================================================
    print("Loading best model for validation...")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Load the best model weights
    model = InterleavedBiGRU()
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare for global validation
    val_preds = []
    val_targets = []
    val_ids = []

    # Identify indices of the scored columns
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    print("Running validation inference...")
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            y = batch["y"].to(device)
            ids = batch["id"]

            # Forward pass
            preds = model(x, pair_indices)

            # Slice to scored length (68)
            preds_scored = preds[:, : Config.SEQ_SCORED, :]

            val_preds.append(preds_scored.cpu())
            val_targets.append(y.cpu())
            val_ids.extend(ids)

    # Concatenate results
    val_preds = torch.cat(val_preds, dim=0)
    val_targets = torch.cat(val_targets, dim=0)

    # Filter to scored columns only
    val_preds_filtered = val_preds[:, :, scored_indices]
    val_targets_filtered = val_targets[:, :, scored_indices]

    # Calculate MCRMSE
    final_metric = mcrmse_loss(val_targets_filtered, val_preds_filtered).item()

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 5. Failure Analysis
    # =========================================================================
    print("\nPerforming Failure Analysis...")

    # Calculate RMSE per sample (averaged over scored columns and positions)
    # Shape: (N, 68, 3) -> Mean Squared Error per sample
    sample_mse = torch.mean(
        (val_targets_filtered - val_preds_filtered) ** 2, dim=(1, 2)
    )
    sample_rmse = torch.sqrt(sample_mse).numpy()

    # Load metadata to get features
    val_meta_df = pd.read_parquet(Config.VAL_DATA_PATH)

    # Ensure alignment by ID
    val_meta_df = val_meta_df.set_index("id").loc[val_ids].reset_index()

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {
            "rmse": sample_rmse,
            "signal_to_noise": val_meta_df["signal_to_noise"].values,
            "SN_filter": val_meta_df["SN_filter"].values,
        }
    )

    # Calculate correlations
    correlations = analysis_df.corr()["rmse"].drop("rmse")
    print("Correlation between Error (RMSE) and Metadata Features:")
    print(correlations)

    # =========================================================================
    # 6. Submission
    # =========================================================================
    THRESHOLD = 0.7247761841173526

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Load test loader
        test_loader = get_test_loader(load_cached_data=True)

        # Run inference (uses the best model saved on disk)
        inference(InterleavedBiGRU, test_loader)

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
