import sys
import os
import torch
import numpy as np
import pandas as pd
import time

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import get_data_loaders
from library.trainer import Trainer, generate_submission


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)

    # Use Config.EPOCHS (60) to allow full convergence for hybrid architecture (Cite 00039).
    pass

    print("=== Configuration ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading
    print("\n=== Loading Data ===")
    # Load cached data to save preprocessing time
    train_loader, val_loader, test_loader, test_ids = get_data_loaders(
        load_cached_data=True
    )

    # 3. Training
    print("\n=== Starting Training ===")
    trainer = Trainer(train_loader, val_loader)
    trainer.fit()

    # 4. Validation Assessment & Failure Analysis
    print("\n=== Validation Assessment & Failure Analysis ===")

    # Load best model for accurate analysis
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}")
        trainer.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        )
    else:
        print("Warning: Best model not found. Using current model state.")

    trainer.model.eval()

    all_preds = []
    all_targets = []
    all_inputs = []

    # Inference loop (No gradients for speed/memory)
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(Config.DEVICE)
            batch_y = batch_y.to(Config.DEVICE)

            # Forward pass
            preds = trainer.model(batch_x)

            # Store results on CPU
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            all_inputs.append(batch_x.cpu().numpy())

    # Concatenate all batches
    # Shapes: (N_breaths, 80) and (N_breaths, 80, Features)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_inputs = np.concatenate(all_inputs, axis=0)

    # Identify u_out feature for masking
    try:
        u_out_idx = Config.FEATURE_COLS.index("u_out")
    except ValueError:
        raise ValueError("'u_out' not found in feature columns.")

    u_out = all_inputs[:, :, u_out_idx]

    # Calculate Metric: Mean Absolute Error on Inspiratory Phase (u_out == 0)
    mask = u_out == 0

    # Flatten arrays based on mask to compute metric over all valid time steps
    masked_preds = all_preds[mask]
    masked_targets = all_targets[mask]

    abs_errors = np.abs(masked_preds - masked_targets)
    final_metric = np.mean(abs_errors)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation of Error with Features
    print("\n--- Failure Analysis ---")
    print("Calculating correlation between input features and error magnitude...")

    # Flatten inputs corresponding to the mask
    # Reshape inputs to (Total_Time_Steps, Features) then apply mask
    flat_inputs = all_inputs.reshape(-1, len(Config.FEATURE_COLS))
    flat_mask = mask.flatten()
    masked_inputs = flat_inputs[flat_mask]

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(masked_inputs, columns=Config.FEATURE_COLS)
    analysis_df["error_magnitude"] = abs_errors

    # Compute correlations
    correlations = analysis_df.corr()["error_magnitude"].sort_values(ascending=False)

    print("Top Feature Correlations with Error:")
    print(correlations)

    # 5. Submission Generation
    print("\n=== Submission Check ===")
    THRESHOLD = 0.22291307151317596

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) < Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(trainer, test_loader, test_ids)
    else:
        print(
            f"Metric ({final_metric}) >= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
