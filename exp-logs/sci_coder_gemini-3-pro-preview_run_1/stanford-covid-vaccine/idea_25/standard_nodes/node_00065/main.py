import sys
import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Ensure we can import from the current directory
sys.path.append(".")

from library.config import Config
from library.dataset import get_dataloaders
from library.engine import Engine
from library.metrics import compute_mcrmse


def main():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration
    # --------------------------------------------------------------------------
    # Set a reasonable number of epochs for a fast baseline
    Config.EPOCHS = 15
    # Setup directories and seeds
    Config.setup()

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\nLoading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # --------------------------------------------------------------------------
    # 3. Training
    # --------------------------------------------------------------------------
    print("\nInitializing Engine...")
    engine = Engine()

    print("Starting Training...")
    # Train the model
    engine.fit(
        train_loader, val_loader, epochs=Config.EPOCHS, early_stopping_patience=5
    )

    # --------------------------------------------------------------------------
    # 4. Final Validation & Metrics
    # --------------------------------------------------------------------------
    print("\nEvaluating Best Model...")

    # Load the best checkpoint to ensure we evaluate the optimal model
    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"Loading weights from {Config.CHECKPOINT_PATH}")
        engine.model.load_state_dict(
            torch.load(Config.CHECKPOINT_PATH, map_location=engine.device)
        )
    else:
        print("Warning: No checkpoint found. Using last model state.")

    # Run inference on validation set
    engine.model.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for seq, loop, dist, targets, mask in val_loader:
            seq = seq.to(engine.device)
            loop = loop.to(engine.device)
            dist = dist.to(engine.device)
            targets = targets.to(engine.device)
            mask = mask.to(engine.device)

            # Forward pass
            preds = engine.model(seq, loop, dist, mask)

            # Move to CPU for metric calculation
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(mask.cpu())

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Compute MCRMSE
    final_metric = compute_mcrmse(all_preds, all_targets, all_masks)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # --------------------------------------------------------------------------
    # 5. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate error per sample for correlation analysis
    # We will use the Mean Squared Error (MSE) per sample over valid positions

    # Expand mask to match predictions: (N, 68) -> (N, 68, 3)
    mask_expanded = all_masks.unsqueeze(-1).expand_as(all_preds).float()

    # Squared Error: (N, 68, 3)
    squared_errors = (all_preds - all_targets) ** 2

    # Mask out invalid positions
    masked_squared_errors = squared_errors * mask_expanded

    # Sum errors per sample: (N,)
    sum_errors = masked_squared_errors.sum(dim=(1, 2))

    # Count valid positions per sample: (N,)
    valid_counts = mask_expanded.sum(dim=(1, 2))
    valid_counts = torch.clamp(valid_counts, min=1.0)  # Avoid div by zero

    # MSE per sample
    mse_per_sample = sum_errors / valid_counts
    # RMSE per sample (scalar metric for quality)
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load validation metadata to get features
    if os.path.exists(Config.VAL_FILE):
        val_df = pd.read_parquet(Config.VAL_FILE)

        # Check alignment
        if len(val_df) == len(rmse_per_sample):
            val_df["model_error"] = rmse_per_sample

            print("Correlations with Model Error (RMSE per sample):")

            # 1. Signal to Noise
            if "signal_to_noise" in val_df.columns:
                corr, _ = pearsonr(val_df["signal_to_noise"], val_df["model_error"])
                print(f"  Error vs Signal_to_Noise: {corr:.4f}")

            # 2. SN Filter
            if "SN_filter" in val_df.columns:
                corr, _ = pearsonr(val_df["SN_filter"], val_df["model_error"])
                print(f"  Error vs SN_filter:       {corr:.4f}")

            # 3. Sequence Composition (e.g., Count of 'A')
            val_df["count_A"] = val_df["sequence"].apply(lambda x: x.count("A"))
            corr, _ = pearsonr(val_df["count_A"], val_df["model_error"])
            print(f"  Error vs Count_A:         {corr:.4f}")

        else:
            print(
                f"Warning: Validation DataFrame length ({len(val_df)}) does not match predictions ({len(rmse_per_sample)}). Skipping correlation."
            )
    else:
        print("Warning: Validation metadata file not found. Skipping failure analysis.")

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    # Threshold defined in task
    SUBMISSION_THRESHOLD = 0.6199890971183777

    if final_metric < SUBMISSION_THRESHOLD:
        print(
            f"\nMetric {final_metric:.6f} < {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        engine.predict(test_loader)
    else:
        print(
            f"\nMetric {final_metric:.6f} >= {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
