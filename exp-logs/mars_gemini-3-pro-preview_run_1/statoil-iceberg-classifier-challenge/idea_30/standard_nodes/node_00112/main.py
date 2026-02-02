import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel

# Import provided library functions
from library.utils import set_seed
from library.data import get_dataloaders, IcebergDataset, get_transforms
from library.model import IcebergResNet18
from library.inference import (
    validate_model,
    predict_with_tta,
    save_submission,
    load_test_ids,
)
from library.training import Trainer


def main():
    # --- 1. Setup & Configuration ---
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    MAX_EPOCHS_VAL = 35
    PATIENCE = 8
    SUBMISSION_THRESHOLD = 0.16918645240183008

    # --- 2. Calibration & Validation Phase ---
    # Cite solution_lesson_node_00040: Global Epoch Selection via Cross-Validation
    print("Initializing Trainer for 5-Fold Calibration...")
    trainer = Trainer(working_dir="./working/idea_30_optimized")

    # Run 5-Fold CV to get robust optimal steps and validation metric
    # Cite solution_lesson_node_00070: Increased max_epochs to 75 to allow convergence with gentle decay
    avg_steps, avg_loss, last_state, last_val_idx, X, a, y = (
        trainer.run_calibration_phase(batch_size=BATCH_SIZE, max_epochs=75)
    )

    # --- 3. Reporting & Analysis ---
    print(f"Final Validation Metric: {avg_loss:.17f}")

    # Failure Analysis on the last fold
    print("\nPerforming Failure Analysis (on last fold)...")

    # Reconstruct Validation Loader for the last fold
    X_val, a_val, y_val = X[last_val_idx], a[last_val_idx], y[last_val_idx]
    val_ds = IcebergDataset(X_val, a_val, y_val, transform=get_transforms("val"))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = IcebergResNet18(dropout_rate=0.5).to(device)
    model.load_state_dict(last_state)

    _, preds, targets = validate_model(model, val_loader, device)

    # Extract angles
    val_angles = a_val  # Already numpy array from Trainer return

    # Calculate absolute error
    errors = np.abs(preds - targets)

    # Compute correlation
    mask = ~np.isnan(val_angles)
    if np.sum(mask) > 0:
        corr = np.corrcoef(errors[mask], val_angles[mask])[0, 1]
        print(f"Correlation between Error and Incidence Angle: {corr:.4f}")

    # --- 4. Production Phase (Full Fit) ---
    if avg_loss < SUBMISSION_THRESHOLD:
        print("\nMetric threshold met. Proceeding to Production Phase...")

        # Cite solution_lesson_node_00049: SWA Ensemble with Global Epoch Selection
        print(f"Optimal Gradient Steps: {avg_steps}")

        # Train 5 SWA Models on Full Data
        swa_model_paths = trainer.run_production_phase(
            optimal_steps=avg_steps, batch_size=BATCH_SIZE, num_models=5
        )

        # Re-initialize test loader for submission
        _, _, test_loader = get_dataloaders(
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
        )

        # --- 5. Inference & Submission ---
        print("\nGenerating Submission...")
        test_ids = load_test_ids()
        ensemble_preds = []

        for path in swa_model_paths:
            # Load SWA Model
            base_model = IcebergResNet18(dropout_rate=0.5).to(device)
            swa_model = AveragedModel(base_model)
            swa_model.load_state_dict(torch.load(path, map_location=device))

            # Predict
            preds = predict_with_tta(swa_model, test_loader, device)
            ensemble_preds.append(preds)

        # Average Predictions
        avg_preds = np.mean(ensemble_preds, axis=0)

        # Save
        save_submission(avg_preds, test_ids, "./submission/submission.csv")

    else:
        print(
            f"Validation metric {best_loss} did not meet threshold {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
