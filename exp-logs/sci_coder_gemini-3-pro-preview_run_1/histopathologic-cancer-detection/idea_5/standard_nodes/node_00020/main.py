import sys
import os
import numpy as np
import torch
import pandas as pd

# Ensure library is in path
sys.path.append("./library")

from library.config import Config
from library.train import run_training
from library.inference import run_inference, load_ensemble_models, TTAEngine
from library.data import get_dataloaders
from library.utils import set_seed, calculate_auc


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    print("--------------------------------------------------")
    print("Step 1: Training Heterogeneous Ensemble")
    print("--------------------------------------------------")
    # Run training with full data (debug=False) to maximize performance.
    # The A100 GPU is capable of processing the 140k 48x48 images efficiently
    # within the time limit even with 20 epochs.
    run_training(debug=False)

    print("\n--------------------------------------------------")
    print("Step 2: Validation & Failure Analysis")
    print("--------------------------------------------------")

    # Load validation data
    print("Loading validation data...")
    dataloaders = get_dataloaders(
        train_path=Config.TRAIN_METADATA_PATH,
        val_path=Config.VAL_METADATA_PATH,
        test_path=Config.TEST_METADATA_PATH,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=False,
    )
    val_loader = dataloaders["val"]

    # Load trained ensemble models
    models = load_ensemble_models(Config.DEVICE)

    # Initialize TTA Engine for validation inference
    # We use TTA for validation to get the most accurate estimate of test performance
    # and to ensure the metric is comparable to the final submission capability.
    engine = TTAEngine(models, Config.DEVICE)

    print("Generating predictions on validation set...")
    val_preds = engine.predict(val_loader)

    # Extract labels and image statistics for failure analysis
    print("Extracting metadata for analysis...")
    val_labels = []
    brightness_vals = []
    contrast_vals = []

    # Iterate through loader to get ground truth and image stats
    # We iterate again because engine.predict consumes the loader but only returns preds.
    # Since shuffle=False for validation, the order is preserved.
    for images, labels in val_loader:
        # images: Tensor (B, C, H, W)
        # labels: Tensor (B,)

        # Append labels
        val_labels.extend(labels.numpy())

        # Calculate simple image stats (on normalized tensors)
        # Brightness ~ Mean intensity across spatial dims and channels
        b_batch = images.mean(dim=(1, 2, 3)).numpy()
        # Contrast ~ Std deviation of intensity across spatial dims and channels
        c_batch = images.std(dim=(1, 2, 3)).numpy()

        brightness_vals.extend(b_batch)
        contrast_vals.extend(c_batch)

    val_labels = np.array(val_labels)
    brightness_vals = np.array(brightness_vals)
    contrast_vals = np.array(contrast_vals)

    # Calculate Validation Metric
    final_auc = calculate_auc(val_labels, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    # Calculate error magnitude
    errors = np.abs(val_labels - val_preds)

    # Calculate correlations
    # using numpy corrcoef, returns matrix [[1, r], [r, 1]]
    corr_bright = np.corrcoef(errors, brightness_vals)[0, 1]
    corr_contrast = np.corrcoef(errors, contrast_vals)[0, 1]

    print(f"Failure Analysis - Error Correlation with Brightness: {corr_bright:.6f}")
    print(f"Failure Analysis - Error Correlation with Contrast: {corr_contrast:.6f}")

    print("\n--------------------------------------------------")
    print("Step 3: Submission Decision")
    print("--------------------------------------------------")

    target_threshold = 0.9849192531860572

    if final_auc > target_threshold:
        print(f"Validation AUC ({final_auc}) exceeds threshold ({target_threshold}).")
        print("Proceeding to Test Inference...")
        run_inference(debug=False)
    else:
        print(
            f"Validation AUC ({final_auc}) does not exceed threshold ({target_threshold})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
