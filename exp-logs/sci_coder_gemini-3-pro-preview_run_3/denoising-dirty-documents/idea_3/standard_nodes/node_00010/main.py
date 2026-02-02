import sys
import os
import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.trainer import Trainer
from library.dataset import get_dataloaders
from library.inference import generate_predictions


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for a fast baseline run as per requirements
    # Reduce epochs and patience to ensure execution within time limits
    Config.NUM_EPOCHS = 5
    Config.EARLY_STOPPING_PATIENCE = 2

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Ensure directories exist
    Config.create_directories()

    print("Configuration set for fast baseline.")
    print(f"Epochs: {Config.NUM_EPOCHS}, Patience: {Config.EARLY_STOPPING_PATIENCE}")

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    trainer = Trainer()

    # Use a debug limit to ensure the training is fast (e.g., 10,000 patches)
    # load_cached_data=True allows using pre-computed npy files if available
    print("Starting training...")
    trainer.fit(load_cached_data=True, debug_limit=10000)

    # ---------------------------------------------------------
    # 3. Validation & Metrics
    # ---------------------------------------------------------
    print("Performing final validation...")

    # Reload the best model checkpoint saved during training to ensure we evaluate the best state
    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded best model checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
        )
    else:
        print("Warning: No checkpoint found. Using current model state.")

    # Get full validation loader (no debug limit for accurate metric calculation)
    _, val_loader = get_dataloaders(load_cached_data=True, debug_limit=None)

    # Compute metric using the trainer's validation logic
    final_rmse = trainer.validate(val_loader)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_rmse}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("Running failure analysis...")
    trainer.model.eval()
    device = trainer.device

    all_inputs = []
    all_targets = []
    all_preds = []

    # Collect predictions and targets for the entire validation set
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass: Predict Noise
            noise_pred = trainer.model(inputs)

            # Reconstruct Clean Image: Input - Noise
            clean_pred = inputs - noise_pred
            clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

            # Store flattened arrays on CPU to avoid OOM during analysis
            all_inputs.append(inputs.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())
            all_preds.append(clean_pred.cpu().numpy().flatten())

    # Concatenate all batches into single arrays
    flat_inputs = np.concatenate(all_inputs)
    flat_targets = np.concatenate(all_targets)
    flat_preds = np.concatenate(all_preds)

    # Calculate Error Magnitude (Absolute Error)
    error_magnitude = np.abs(flat_preds - flat_targets)

    # Calculate correlations
    # 1. Error vs Input Intensity (Are bright/dark pixels harder to clean?)
    if len(error_magnitude) > 1:
        corr_input, _ = pearsonr(error_magnitude, flat_inputs)
        corr_target, _ = pearsonr(error_magnitude, flat_targets)
    else:
        corr_input, corr_target = 0.0, 0.0

    print(f"Correlation (Error Magnitude vs Input Intensity): {corr_input}")
    print(f"Correlation (Error Magnitude vs Target Intensity): {corr_target}")

    # ---------------------------------------------------------
    # 5. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.016654925420880318

    if final_rmse < THRESHOLD:
        print(
            f"Validation metric {final_rmse} is better than threshold {THRESHOLD}. Generating submission..."
        )
        generate_predictions()
    else:
        print(
            f"Validation metric {final_rmse} is not better than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
