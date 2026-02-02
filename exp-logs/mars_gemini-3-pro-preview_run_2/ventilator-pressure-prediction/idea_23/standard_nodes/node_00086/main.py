import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr

# Ensure library modules are importable
sys.path.append(os.getcwd())

from library.config import Config
from library.train import Trainer
from library.model import WPABiLSTM
from library.inference import predict as inference_predict


def main():
    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    print("--- Configuring Optimized Training Run ---")
    # Using Config defaults (100 epochs) to allow long-tail convergence (Cite Lesson 00029)

    # Initialize Trainer (this loads datasets and sets up model/optimizer)
    trainer = Trainer()

    # Limit Training Data
    # Total breaths ~54k. Using 40,000 breaths to balance coverage and runtime.
    # Each breath is 80 steps.
    BREATHS_TO_USE = 40000
    SEQ_LEN = 80
    limit_samples = BREATHS_TO_USE * SEQ_LEN

    if len(trainer.train_dataset) > limit_samples:
        print(
            f"Subsetting training data to {BREATHS_TO_USE} breaths ({limit_samples} steps)..."
        )
        indices = list(range(limit_samples))
        subset_train = Subset(trainer.train_dataset, indices)

        # Re-create the loader with the subset
        trainer.train_loader = DataLoader(
            subset_train,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    # Train
    print("--- Starting Training ---")
    trainer.fit()

    # Validation & Failure Analysis
    print("\n--- Starting Validation & Failure Analysis ---")
    device = torch.device(Config.DEVICE)
    model = WPABiLSTM().to(device)

    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("Error: Best model not found.")
        return

    print(f"Loading best model from {Config.BEST_MODEL_PATH}...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    val_loader = trainer.val_loader

    all_preds = []
    all_targets = []
    all_inputs = []

    # Inference on Validation Set
    print("Running inference on validation set...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_inputs.append(inputs.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_inputs = torch.cat(all_inputs, dim=0)

    # Calculate Metric: MAE on Inspiratory Phase (u_out == 0)
    # Find u_out index
    try:
        u_out_idx = len(Config.CONTINUOUS_FEATURES) + Config.BINARY_FEATURES.index(
            "u_out"
        )
    except ValueError:
        u_out_idx = -1

    u_out = all_inputs[:, :, u_out_idx]
    mask = u_out == 0

    abs_error = torch.abs(all_preds - all_targets)
    insp_error = abs_error[mask]

    final_metric = insp_error.mean().item()
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation of Error with Features
    print("\n--- Failure Analysis: Error Correlations (Inspiratory Phase) ---")
    flat_error = insp_error.numpy()

    feature_names = Config.CONTINUOUS_FEATURES + Config.BINARY_FEATURES

    # Iterate through features to calculate correlation
    for i, feat_name in enumerate(feature_names):
        # Extract feature values for the inspiratory phase
        feat_vals = all_inputs[:, :, i][mask].numpy()

        # Check for constant values to avoid warnings (and NaNs)
        if np.std(feat_vals) > 1e-9:
            corr, _ = pearsonr(flat_error, feat_vals)
            print(f"Correlation with {feat_name}: {corr:.6f}")
        else:
            print(f"Correlation with {feat_name}: NaN (Constant)")

    # Submission Logic
    THRESHOLD = 0.1619843989610672
    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        inference_predict()
    else:
        print(
            f"\nMetric {final_metric} does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
