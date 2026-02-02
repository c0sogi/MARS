import os
import sys
import pandas as pd
import numpy as np
import torch

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import SEED, DEVICE, CHECKPOINT_DIR, TRAIN_METADATA, VAL_METADATA
from library.train import run_training
from library.inference import predict_and_submit
from library.dataset import get_dataloaders
from library.model import ParallelDilatedCNN
from library.utils import set_seed, calculate_fbeta


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Training Phase
    # run_training handles the loop, saving the best model to CHECKPOINT_DIR/best_model.pth
    # We use the default configuration (20 epochs, batch size 32) which is fast for the small dataset.
    print("Starting training...")
    run_training()

    # 3. Validation & Failure Analysis Phase
    print("Starting validation and failure analysis...")

    # Load the best model
    model = ParallelDilatedCNN().to(DEVICE)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found. Training may have failed.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    # Get Validation Data
    # We force load_cached_data=True to utilize any cache generated during training
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    if len(val_loader) == 0:
        print("Error: Validation loader is empty.")
        return

    all_preds = []
    all_targets = []
    sample_errors = []

    # Inference loop (No Grad)
    with torch.no_grad():
        for data, target, sample_ids in val_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            # Forward pass
            logits = model(data)
            probs = torch.sigmoid(logits)

            # Move to CPU for metric calculation
            probs_cpu = probs.cpu()
            target_cpu = target.cpu()

            all_preds.append(probs_cpu)
            all_targets.append(target_cpu)

            # Calculate Mean Absolute Error per sample for Failure Analysis
            # Shape: (Batch, 1, H, W) -> Mean over (1, H, W)
            mae = torch.abs(probs_cpu - target_cpu).mean(dim=(1, 2, 3)).numpy()

            for sid, err in zip(sample_ids, mae):
                sample_errors.append({"sample_id": sid, "error": err})

    # Concatenate results
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Calculate Final Validation Metric (F0.5)
    # Using threshold 0.5 as standard for the uncalibrated metric reporting
    bin_preds = (all_preds > 0.5).float()
    final_metric = calculate_fbeta(bin_preds, all_targets, beta=0.5)

    # Print exactly as required
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation with Metadata
    print("Performing Failure Analysis...")
    if len(sample_errors) > 0:
        error_df = pd.DataFrame(sample_errors)
        val_metadata = val_loader.dataset.metadata

        # Merge error data with metadata features (x, y)
        analysis_df = pd.merge(error_df, val_metadata, on="sample_id")

        # Calculate correlations
        features_to_check = ["x", "y"]
        print("Correlation between Error Magnitude and Input Features:")
        for feature in features_to_check:
            if feature in analysis_df.columns:
                corr = analysis_df[feature].corr(analysis_df["error"])
                print(f"  {feature}: {corr:.4f}")

                # Simple interpretation
                if abs(corr) > 0.2:
                    direction = "positive" if corr > 0 else "negative"
                    print(f"    -> Significant {direction} correlation detected.")

    # 4. Submission Phase
    # Generate submission only if metric exceeds threshold
    THRESHOLD = 0.41758
    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric:.5f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        # predict_and_submit handles threshold calibration internally
        predict_and_submit(checkpoint_path=checkpoint_path)
    else:
        print(
            f"Metric ({final_metric:.5f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
