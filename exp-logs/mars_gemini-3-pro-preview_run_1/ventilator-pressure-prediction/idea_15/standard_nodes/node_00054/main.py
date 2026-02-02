import sys
import os
import torch
import numpy as np
import pandas as pd

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.engine import Trainer
from library.utils import seed_everything, ensure_dir


def main():
    # 1. Configuration and Setup
    # Initialize config with debug=False to use the full dataset for high performance
    config = Config(debug=False)

    # Override EPOCHS to 15 to ensure execution completes quickly (Fast Baseline)
    # The A100 GPU is fast enough to handle 15 epochs on this dataset in minutes.
    config.EPOCHS = 15

    # Set seeds for reproducibility
    seed_everything(config.SEED)

    print(f"Initializing Trainer with {config.EPOCHS} epochs...")
    trainer = Trainer(config)

    # 2. Training
    # Runs training loop and saves the best model to config.MODEL_PATH
    trainer.fit()

    # 3. Validation & Failure Analysis
    print("\n=== Starting Validation & Failure Analysis ===")

    # Load the best model for analysis
    if os.path.exists(config.MODEL_PATH):
        print(f"Loading best model from {config.MODEL_PATH}...")
        trainer.model.load_state_dict(
            torch.load(config.MODEL_PATH, map_location=trainer.device)
        )
    else:
        print("Warning: No model checkpoint found. Using current model state.")

    model = trainer.model
    model.eval()
    device = trainer.device
    val_loader = trainer.val_loader

    all_preds = []
    all_targets = []
    all_u_out = []
    all_inputs = []

    # Run inference on validation set
    with torch.no_grad():
        for batch in val_loader:
            x = batch["input"].to(device)
            u_out = batch["u_out"].to(device)
            y = batch["target"].to(device)

            # Forward pass
            output = model(x)
            pred = output["prediction"]

            # Collect results (move to CPU)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_u_out.append(u_out.cpu().numpy())
            all_inputs.append(x.cpu().numpy())

    # Concatenate all batches
    # Shapes: (N_breaths * 80, ) after flattening
    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    u_out = np.concatenate(all_u_out).flatten()

    # Inputs: (N_breaths, 80, N_features) -> Flatten to (N_total, N_features)
    inputs = np.concatenate(all_inputs)
    inputs_flat = inputs.reshape(-1, inputs.shape[-1])

    # Filter for Inspiratory Phase (u_out == 0)
    # The metric is only defined for the inspiratory phase
    insp_mask = u_out == 0

    preds_insp = preds[insp_mask]
    targets_insp = targets[insp_mask]
    inputs_insp = inputs_flat[insp_mask]

    # Calculate Metric (MAE)
    mae = np.mean(np.abs(preds_insp - targets_insp))
    print(f"Final Validation Metric: {mae:.16f}")

    # Failure Analysis: Correlation between Error Magnitude and Features
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    errors = np.abs(preds_insp - targets_insp)
    feature_names = config.FEATURE_LIST

    for i, feat_name in enumerate(feature_names):
        feat_vals = inputs_insp[:, i]

        # Calculate correlation (handle constant features)
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(errors, feat_vals)[0, 1]

        print(f"  {feat_name}: {corr:.4f}")

    # 4. Submission
    # Generate submission only if metric is below threshold
    threshold = 0.2164510190486908

    if mae < threshold:
        print(f"\nValidation metric {mae:.6f} < {threshold}. Generating submission...")
        trainer.generate_submission()
    else:
        print(f"\nValidation metric {mae:.6f} >= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
