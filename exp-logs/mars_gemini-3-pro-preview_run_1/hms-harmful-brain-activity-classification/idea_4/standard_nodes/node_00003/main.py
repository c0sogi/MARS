import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.train import run_training
from library.inference import run_inference


def main():
    # 1. Setup & Configuration
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Configure for a fast baseline run
    # We use the full dataset for 1 epoch to get a representative model within the time limit.
    Config.EPOCHS = 1
    Config.DEBUG = False

    # Ensure directories exist
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("Starting Fast Baseline Orchestration...")
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Debug={Config.DEBUG}, Device={device}"
    )

    # 2. Training
    # run_training handles model initialization, training loop, and saving the best model.
    model = run_training(debug=Config.DEBUG, epochs=Config.EPOCHS)

    # 3. Full Validation
    print("\nPerforming Final Validation on Hold-out Set...")

    # Get the full validation loader (debug=False ensures full set)
    # We use a larger batch size for inference speed
    _, val_loader, _ = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE * 2,
        debug=False,
    )

    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            eeg, spec = inputs
            eeg = eeg.to(device)
            spec = spec.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model((eeg, spec))

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    # Concatenate results
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute Final Metric (KL Divergence)
    # F.kl_div expects log_probs as input and probs as target
    log_probs = F.log_softmax(all_logits, dim=1)
    final_metric = F.kl_div(log_probs, all_targets, reduction="batchmean").item()

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Compute per-sample KL divergence (reduction='none' then sum over classes)
    # Shape: (N_samples,)
    per_sample_loss = (
        F.kl_div(log_probs, all_targets, reduction="none").sum(dim=1).numpy()
    )

    # Retrieve metadata from the dataset
    val_meta = val_loader.dataset.metadata.copy()

    # Sanity check alignment
    if len(val_meta) == len(per_sample_loss):
        val_meta["error"] = per_sample_loss

        # Calculate correlations
        features_to_check = [
            "eeg_label_offset_seconds",
            "spectrogram_label_offset_seconds",
        ]

        for feature in features_to_check:
            if feature in val_meta.columns:
                # Fill NaNs with 0 for correlation calculation
                feature_values = val_meta[feature].fillna(0).values
                corr, _ = pearsonr(val_meta["error"].values, feature_values)
                print(f"Correlation between Error and {feature}: {corr}")
    else:
        print(
            f"Warning: Metadata length ({len(val_meta)}) matches predictions ({len(per_sample_loss)}) mismatch. Skipping detailed analysis."
        )

    # 5. Submission Logic
    THRESHOLD = 0.9343236597211053

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        # Run inference on test set
        run_inference(debug=False, batch_size=Config.BATCH_SIZE * 2)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
