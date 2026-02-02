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
from library.data import get_dataloaders
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
    # We perform a single-fold validation run to:
    # a) Determine the optimal gradient step volume (Calibration).
    # b) Calculate the hold-out validation metric (Requirement).
    # c) Perform failure analysis (Requirement).

    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
    )

    print(f"\nStarting Calibration Training (Max Epochs: {MAX_EPOCHS_VAL})...")
    model = IcebergResNet18(dropout_rate=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS_VAL):
        model.train()
        running_loss = 0.0
        samples = 0

        for batch in train_loader:
            images, angles, labels = batch
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Label Smoothing (0.05)
            targets = labels * 0.95 + 0.025

            optimizer.zero_grad()
            outputs = model(images, angles)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            samples += images.size(0)

        train_loss = running_loss / samples

        # Validation
        val_loss, _, _ = validate_model(model, val_loader, device)
        scheduler.step(val_loss)

        # print(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # --- 3. Reporting & Analysis ---
    print(f"Final Validation Metric: {best_loss:.17f}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.load_state_dict(best_model_state)
    _, preds, targets = validate_model(model, val_loader, device)

    # Extract angles from validation set for correlation analysis
    val_angles = []
    for batch in val_loader:
        _, angles, _ = batch
        val_angles.append(angles.numpy())
    val_angles = np.concatenate(val_angles).flatten()

    # Calculate absolute error
    errors = np.abs(preds - targets)

    # Compute correlation
    # Filter out any potential NaNs (though preprocessing handles this)
    mask = ~np.isnan(val_angles)
    if np.sum(mask) > 0:
        corr = np.corrcoef(errors[mask], val_angles[mask])[0, 1]
        print(f"Correlation between Error and Incidence Angle: {corr:.4f}")

    # --- 4. Production Phase (Full Fit) ---
    if best_loss < SUBMISSION_THRESHOLD:
        print("\nMetric threshold met. Proceeding to Production Phase...")

        # Calculate optimal gradient steps
        steps_per_epoch = len(train_loader)
        optimal_steps = best_epoch * steps_per_epoch
        print(f"Optimal Gradient Steps: {optimal_steps}")

        # Initialize Trainer for Full-Fit Strategy
        # We use a separate working directory to avoid cache conflicts if any
        trainer = Trainer(working_dir="./working/production_run")

        # Train 5 SWA Models on Full Data
        swa_model_paths = trainer.run_production_phase(
            optimal_steps=optimal_steps, batch_size=BATCH_SIZE, num_models=5
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
