import os
import numpy as np
import pandas as pd
import torch
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders
from library.trainer import Trainer


def main():
    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    SEED = 42
    # Increased batch size for A100 efficiency and speed
    BATCH_SIZE = 512
    # Extended epochs for full convergence (Cite solution_lesson_node_00005)
    EPOCHS = 150
    LEARNING_RATE = 1e-3
    HIDDEN_DIM = 512
    NUM_LAYERS = 4
    INPUT_DIM = 14
    CACHE_DIR = "./working/idea_4"
    INPUT_DIR = "./input"
    THRESHOLD = 0.26559123396873474

    print("Starting runfile execution...")
    seed_everything(SEED)
    device = get_device()

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    # Initialize Trainer with optimized parameters
    trainer = Trainer(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    # Fit model (handles data loading, feature engineering, and caching internally)
    trainer.fit(data_dir=INPUT_DIR, cache_dir=CACHE_DIR)

    # ---------------------------------------------------------
    # 3. Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\nStarting Validation and Failure Analysis...")

    # Reload validation loader (cached, so this is fast)
    _, val_loader, _ = get_dataloaders(
        data_dir=INPUT_DIR,
        batch_size=BATCH_SIZE,
        load_cached_data=True,
        cache_dir=CACHE_DIR,
    )

    trainer.model.eval()

    all_preds = []
    all_targets = []
    all_u_out = []
    all_inputs = []

    # Inference loop on validation set
    with torch.no_grad():
        for batch in val_loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            # Forward pass
            preds = trainer.model(X)

            # Collect data (move to CPU to save GPU memory)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
            all_u_out.append(u_out.cpu())
            all_inputs.append(X.cpu())

    # Concatenate all batches
    # Shapes: (N_breaths, 80, 1) -> flatten to (N_steps,)
    y_pred_flat = torch.cat(all_preds).squeeze(-1).numpy().flatten()
    y_true_flat = torch.cat(all_targets).numpy().flatten()
    u_out_flat = torch.cat(all_u_out).numpy().flatten()

    # Inputs: (N_breaths, 80, 14) -> flatten to (N_steps, 14)
    X_flat = torch.cat(all_inputs).numpy().reshape(-1, INPUT_DIM)

    # Calculate Metric (MAE on Inspiratory Phase: u_out == 0)
    insp_mask = u_out_flat == 0

    if np.sum(insp_mask) > 0:
        final_metric = np.mean(np.abs(y_pred_flat[insp_mask] - y_true_flat[insp_mask]))
    else:
        final_metric = 0.0

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation of Error with Features
    # We analyze error magnitude
    errors = np.abs(y_pred_flat - y_true_flat)

    # Filter for inspiratory phase for analysis (since that's what we care about)
    errors_insp = errors[insp_mask]
    X_insp = X_flat[insp_mask]

    # Feature names matching the order in library/dataset.py
    feature_names = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_diff1",
        "u_in_diff2",
        "u_in_cumsum",
        "R_u_in",
        "vol_C",
    ]

    print("\nFailure Analysis - Feature Correlations with Error (Inspiratory Phase):")
    for i, name in enumerate(feature_names):
        if i < X_insp.shape[1]:
            feat_values = X_insp[:, i]
            # Check for constant values to avoid NaN correlation
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(errors_insp, feat_values)[0, 1]
                print(f"{name}: {corr:.4f}")
            else:
                print(f"{name}: NaN (Constant)")

    # ---------------------------------------------------------
    # 4. Submission
    # ---------------------------------------------------------
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        trainer.predict(data_dir=INPUT_DIR, cache_dir=CACHE_DIR)
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
