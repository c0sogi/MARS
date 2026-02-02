import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

# Import library modules
from library.config import Config
from library.utils import seed_everything, kl_divergence_score
from library.dataset import get_dataloader
from library.model import EEGNet
from library.train import train
from library.inference import predict


def run():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline Execution
    # We limit epochs and training data size to ensure completion within the time limit
    # while still allowing the model to learn enough to be evaluated.
    Config.EPOCHS = 6
    Config.BATCH_SIZE = 32

    # Define subset sizes
    # 20,000 samples is a reasonable subset for a fast baseline on this hardware
    TRAIN_SUBSET_SIZE = 20000

    print(f"Configuration Overrides:")
    print(f"  EPOCHS: {Config.EPOCHS}")
    print(f"  TRAIN_SUBSET_SIZE: {TRAIN_SUBSET_SIZE}")

    # ==========================================
    # 2. Training
    # ==========================================
    print("\n" + "=" * 40)
    print("STARTING TRAINING PHASE")
    print("=" * 40)

    # Train the model
    # This function handles the training loop, validation monitoring, and saving the best model
    # We pass the subset size to speed up the process
    train(debug_subset_size=TRAIN_SUBSET_SIZE)

    # ==========================================
    # 3. Validation & Metrics
    # ==========================================
    print("\n" + "=" * 40)
    print("STARTING VALIDATION ASSESSMENT")
    print("=" * 40)

    device = torch.device(Config.DEVICE)

    # Load Validation Data (Full Set)
    # We use the full validation set for the official metric calculation
    val_loader = get_dataloader(
        mode="val",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        debug_subset=None,
    )

    # Load the Best Model
    model = EEGNet(pretrained=False)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    print(f"Loading best model from {Config.MODEL_PATH}")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    # Inference Loop (No Gradients for speed and memory efficiency)
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            # Targets are needed for metric calculation, keep them on CPU/Tensor

            # Forward pass
            outputs = model(images)

            all_preds.append(outputs.cpu())
            all_targets.append(targets)

    # Concatenate all batches
    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    # Compute Final Metric (KL Divergence)
    # Using the utility function provided in library which handles log-prob conversion
    final_metric = kl_divergence_score(preds_tensor, targets_tensor)

    # Print exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # Calculate KL Divergence per sample for analysis
    # F.kl_div with reduction='none' returns element-wise loss
    # We sum over classes (dim=1) to get loss per sample
    epsilon = 1e-6
    preds_clamped = torch.clamp(preds_tensor, min=epsilon, max=1.0)

    # KL(P || Q) = sum(P(x) * log(P(x)/Q(x)))
    # PyTorch KLDivLoss input is log(Q) (prediction), target is P (truth)
    # Loss = P * (log(P) - log(Q))
    kl_elementwise = F.kl_div(
        torch.log(preds_clamped), targets_tensor, reduction="none"
    )
    kl_per_sample = kl_elementwise.sum(dim=1).numpy()

    # Load Validation Metadata
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment (safety check)
    if len(val_df) != len(kl_per_sample):
        print(
            f"Warning: Metadata length ({len(val_df)}) != Predictions length ({len(kl_per_sample)})"
        )
        min_len = min(len(val_df), len(kl_per_sample))
        val_df = val_df.iloc[:min_len]
        kl_per_sample = kl_per_sample[:min_len]

    # Add error to dataframe
    val_df["error_magnitude"] = kl_per_sample

    # Correlations
    # We look for correlations between the error and metadata features
    feature_cols = [
        "total_votes",
        "eeg_label_offset_seconds",
        "spectogram_label_offset_seconds",
    ]
    existing_cols = [c for c in feature_cols if c in val_df.columns]

    print("Correlation between Error Magnitude and Metadata Features:")
    correlations = (
        val_df[existing_cols + ["error_magnitude"]]
        .corr()["error_magnitude"]
        .drop("error_magnitude")
    )
    print(correlations)

    # ==========================================
    # 5. Submission
    # ==========================================
    print("\n" + "=" * 40)
    print("SUBMISSION GENERATION")
    print("=" * 40)

    THRESHOLD = 1.0081

    if final_metric < THRESHOLD:
        print(
            f"Validation metric ({final_metric}) is better than threshold ({THRESHOLD})."
        )
        print("Generating submission file...")
        # Predict on full test set
        predict(debug_subset_size=None)
    else:
        print(
            f"Validation metric ({final_metric}) did not meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
