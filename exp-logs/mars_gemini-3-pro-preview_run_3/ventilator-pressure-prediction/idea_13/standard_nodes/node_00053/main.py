import os
import sys
import numpy as np
import pandas as pd
import torch

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import prepare_datasets
from library.train import train_model
from library.model import TAPINNet


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Extended training budget for hybrid convergence (Cite Lesson 39).
    # Previous run with 8 epochs underfit (Val MAE ~0.57).
    Config.EPOCHS = 50

    print(f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # 2. Training
    # train_model handles data preparation (caching) and the training loop
    print("\n=== Starting Training ===")
    # Force cache invalidation to ensure u_out scaling fix is applied (Cite debug_lesson_4)
    model = train_model(epochs=Config.EPOCHS, load_cached_data=False, save_model=True)

    # 3. Validation and Failure Analysis
    print("\n=== Starting Validation & Failure Analysis ===")
    device = get_device()
    model.to(device)
    model.eval()

    # Retrieve data loaders (will load from cache generated during training)
    # We need the validation loader specifically
    _, val_loader, test_loader = prepare_datasets(load_cached_data=True)

    val_preds = []
    val_targets = []
    val_inputs = []

    # Validation Inference
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            out = model(x)

            # Collect data for analysis (move to CPU to save GPU memory)
            val_preds.append(out.cpu().numpy())
            val_targets.append(y.cpu().numpy())
            val_inputs.append(x.cpu().numpy())

    # Concatenate all batches
    val_preds = np.concatenate(val_preds)  # Shape: (N_samples, 80)
    val_targets = np.concatenate(val_targets)  # Shape: (N_samples, 80)
    val_inputs = np.concatenate(val_inputs)  # Shape: (N_samples, 80, N_features)

    # Flatten for metric calculation and analysis
    preds_flat = val_preds.flatten()
    targets_flat = val_targets.flatten()
    inputs_flat = val_inputs.reshape(-1, val_inputs.shape[-1])

    # Extract u_out for masking (Index 1 in MODEL_FEATURES)
    # u_out == 0 is Inspiratory phase (what we score)
    u_out_col_idx = Config.MODEL_FEATURES.index("u_out")
    u_out_flat = inputs_flat[:, u_out_col_idx]

    # Create mask: u_out < 0.5 (robust for float)
    insp_mask = u_out_flat < 0.5

    # Calculate Metric
    # Filter only inspiratory phase
    preds_insp = preds_flat[insp_mask]
    targets_insp = targets_flat[insp_mask]

    final_mae = np.mean(np.abs(preds_insp - targets_insp))

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {final_mae}")

    # Failure Analysis
    print("\n--- Failure Analysis (Correlation with Error) ---")
    # Calculate absolute error for inspiratory phase
    errors_insp = np.abs(preds_insp - targets_insp)
    inputs_insp = inputs_flat[insp_mask]

    feature_names = Config.MODEL_FEATURES

    for i, feat_name in enumerate(feature_names):
        feat_vals = inputs_insp[:, i]

        # Calculate Pearson correlation
        if np.std(feat_vals) < 1e-9:
            corr = 0.0  # Constant feature has 0 correlation
        else:
            corr = np.corrcoef(feat_vals, errors_insp)[0, 1]

        print(f"{feat_name}: {corr:.4f}")

    # 4. Submission
    threshold = 0.1642141044139862

    if final_mae < threshold:
        print(
            f"\nMetric {final_mae} is better than threshold {threshold}. Generating submission..."
        )

        test_preds = []

        with torch.no_grad():
            for x in test_loader:
                x = x.to(device)
                out = model(x)
                test_preds.append(out.cpu().numpy())

        # Flatten predictions
        test_preds_flat = np.concatenate(test_preds).flatten()

        # Load Test IDs
        test_ids = np.load(Config.TEST_IDS)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "pressure": test_preds_flat})

        # Save
        os.makedirs("./submission", exist_ok=True)
        sub_path = "./submission/submission.csv"
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric {final_mae} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
