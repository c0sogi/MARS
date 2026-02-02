import sys
import os
import torch
import pandas as pd
import numpy as np

# Import provided library modules
from library.config import Config
from library.utils import set_seed, calculate_roc_auc
from library.data import get_dataloaders
from library.model import SpatialDifferenceCNN
from library.train import run_training
from library.inference import predict_test_set


def main():
    # --- 1. Setup & Configuration ---
    # Adjust Config for A100 GPU usage and fast baseline execution
    Config.BATCH_SIZE = 256
    Config.NUM_WORKERS = 4

    # Set random seeds for reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # --- 2. Training ---
    print("Starting training...")
    # Run training for 5 epochs to ensure completion within time limits
    run_training(debug=False, epochs=5, patience=3)

    # --- 3. Validation & Failure Analysis ---
    print("Starting validation and failure analysis...")

    # Load the best model saved during training
    model = SpatialDifferenceCNN().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Get validation dataloader
    # Note: We re-create the loader here to ensure we iterate over the full validation set
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    all_targets = []
    all_preds = []

    # Lists to store input features for failure analysis
    # We will analyze the statistical properties of the processed "Difference Maps"
    feat_means = []
    feat_stds = []
    feat_maxs = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            # targets are needed for metric calculation

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Collect predictions and targets
            all_targets.extend(targets.numpy())
            all_preds.extend(probs.cpu().numpy().flatten())

            # Extract features from input images for failure analysis
            # images shape: (B, 1, H, W) -> flatten to (B, H*W)
            flat_images = images.view(images.size(0), -1)

            # Calculate stats on GPU then move to CPU
            feat_means.extend(flat_images.mean(dim=1).cpu().numpy())
            feat_stds.extend(flat_images.std(dim=1).cpu().numpy())
            feat_maxs.extend(flat_images.max(dim=1).values.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # Calculate and print Final Validation Metric
    final_auc = calculate_roc_auc(all_targets, all_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation between Error and Input Features
    errors = np.abs(all_targets - all_preds)

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "input_mean": feat_means,
            "input_std": feat_stds,
            "input_max": feat_maxs,
        }
    )

    print("Failure Analysis - Correlation with Error Magnitude:")
    corr_mean = analysis_df["error"].corr(analysis_df["input_mean"])
    corr_std = analysis_df["error"].corr(analysis_df["input_std"])
    corr_max = analysis_df["error"].corr(analysis_df["input_max"])

    print(f"  Input Mean: {corr_mean}")
    print(f"  Input Std:  {corr_std}")
    print(f"  Input Max:  {corr_max}")

    # --- 4. Submission ---
    print("Generating submission for test set...")
    predict_test_set(debug=False)

    print("Process complete.")


if __name__ == "__main__":
    main()
