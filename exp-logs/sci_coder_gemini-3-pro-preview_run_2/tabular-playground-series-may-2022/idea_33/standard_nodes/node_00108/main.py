import sys
import os
import torch
import numpy as np
import warnings

# Ensure local library modules can be imported
sys.path.append(".")

import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
import library.train as train


def main():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration
    # --------------------------------------------------------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seeds for reproducibility
    utils.seed_everything(config.RANDOM_STATE)

    # Hyperparameters for Fast Baseline
    # We use 20 epochs which is sufficient for convergence on this dataset
    # while being significantly faster than the default 40.
    # We increase batch size to 2048 to fully utilize the A100 GPU.
    FAST_EPOCHS = 20
    FAST_BATCH_SIZE = 2048

    print(
        f"Starting Fast Baseline Run: {FAST_EPOCHS} Epochs, {FAST_BATCH_SIZE} Batch Size"
    )

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    # run_training returns the best validation AUC achieved
    best_auc = train.run_training(
        epochs=FAST_EPOCHS, batch_size=FAST_BATCH_SIZE, load_cached_data=True
    )

    # --------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Starting Validation & Failure Analysis ---")

    device = torch.device(config.DEVICE)

    # Load the best model saved during training
    net = model.ManufacturingNet()
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    net.to(device)
    net.eval()

    # Get Validation DataLoader
    # We use the same batch size as training for efficiency
    _, val_loader, _ = data.get_dataloaders(
        batch_size=FAST_BATCH_SIZE, load_cached_data=True
    )

    # Containers for analysis
    all_targets = []
    all_preds = []
    all_cont_features = []

    # Inference Loop
    with torch.no_grad():
        for batch in val_loader:
            # Move data to GPU
            cont = batch["continuous"].to(device, non_blocking=True)
            cat = batch["categorical"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            # Forward pass
            logits = net(cont, cat)

            # Average probabilities across Multi-Sample Dropout heads
            probs = torch.sigmoid(logits).mean(dim=1)

            # Store results on CPU
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())
            all_cont_features.append(cont.cpu().numpy())

    # Concatenate batches
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    X_cont = np.concatenate(all_cont_features)

    # Compute Final Metric
    final_metric = utils.compute_auc(y_true, y_pred)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Calculate absolute error magnitude
    errors = np.abs(y_true - y_pred)

    # Identify continuous feature names (f_00 to f_30, excluding f_27)
    cont_col_names = [f"f_{i:02d}" for i in range(31) if i != 27]

    feature_corrs = []
    # Calculate correlation between Error and each Feature
    for i, col_name in enumerate(cont_col_names):
        if i < X_cont.shape[1]:
            # Pearson correlation [0, 1] is the correlation coefficient
            corr = np.corrcoef(errors, X_cont[:, i])[0, 1]
            feature_corrs.append((col_name, corr))

    # Sort by magnitude of correlation (descending)
    feature_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nFailure Analysis - Top Feature Correlations with Error Magnitude:")
    for name, corr in feature_corrs[:5]:
        print(f"{name}: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 4. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9972336610045187

    if final_metric > THRESHOLD:
        print(
            f"\nMetric {final_metric} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        train.generate_submission(batch_size=FAST_BATCH_SIZE, load_cached_data=True)
    else:
        print(
            f"\nMetric {final_metric} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
