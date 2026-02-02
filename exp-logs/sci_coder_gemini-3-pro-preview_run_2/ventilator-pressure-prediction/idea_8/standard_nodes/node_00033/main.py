import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, WeightedL1Loss, compute_metric
from library.data_processing import prepare_datasets
from library.model import GIDBiLSTM
from library.engine import Trainer


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline execution
    # We reduce epochs and use a subset of data to ensure < 2 hours runtime
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 512  # A100 40GB can handle larger batches for speed

    # Initialize environment
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting execution on device: {device}")

    # ==========================================
    # 2. Data Preparation
    # ==========================================
    print("Preparing datasets...")
    # Load full datasets (will use cache in ./working/idea_8 if available)
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        debug=False, load_cached_data=True
    )

    # Create a 50% subset of training data for the fast baseline
    # This ensures we respect the "Limit maximum number of training samples" constraint
    # while providing enough data for convergence.
    subset_size = int(len(train_dataset) * 0.5)
    subset_indices = torch.randperm(len(train_dataset))[:subset_size]
    train_subset = Subset(train_dataset, subset_indices)

    print(
        f"Training Data: {len(train_subset)} breaths (Subsampled from {len(train_dataset)})"
    )
    print(f"Validation Data: {len(val_dataset)} breaths (Full set)")

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing model...")
    model = GIDBiLSTM().to(device)

    criterion = WeightedL1Loss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    trainer = Trainer(model, device, optimizer, scheduler, criterion)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_val_mae = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = trainer.train_one_epoch(train_loader)

        # Validate
        val_loss, val_mae = trainer.validate(val_loader)

        # Step Scheduler
        if scheduler:
            scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val MAE: {val_mae:.5f}"
        )

        # Save Best Model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)

    print(f"Training complete. Best Val MAE: {best_val_mae}")

    # ==========================================
    # 5. Final Validation & Failure Analysis
    # ==========================================
    print("\n--- Final Validation & Failure Analysis ---")

    # Load the best model checkpoint
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))
    model.eval()

    # Collect predictions, targets, and inputs from the full validation set
    val_preds_list = []
    val_targets_list = []
    val_inputs_list = []
    val_u_out_list = []

    with torch.no_grad():
        for batch in val_loader:
            X = batch[0].to(device)
            u_out = batch[1].to(device)
            y = batch[2].to(device)

            preds = model(X)

            val_preds_list.append(preds.cpu().numpy())
            val_targets_list.append(y.cpu().numpy())
            val_u_out_list.append(u_out.cpu().numpy())
            val_inputs_list.append(X.cpu().numpy())

    # Concatenate all batches
    val_preds = np.concatenate(val_preds_list)
    val_targets = np.concatenate(val_targets_list)
    val_u_out = np.concatenate(val_u_out_list)
    val_inputs = np.concatenate(val_inputs_list)

    # Compute Final Metric (Full Precision)
    final_metric = compute_metric(
        val_preds.flatten(), val_targets.flatten(), val_u_out.flatten()
    )
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Calculate absolute error per time step
    abs_error = np.abs(val_preds - val_targets)

    # Filter for Inspiratory Phase (u_out < 0.5)
    # We only care about errors in the inspiratory phase as per the metric
    insp_mask = val_u_out < 0.5

    error_flat = abs_error[insp_mask]
    inputs_flat = val_inputs[insp_mask]

    # Feature names corresponding to the order in data_processing.py
    feature_names = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "dt",
        "u_in_cumsum",
        "R_u_in",
        "u_in_cumsum_C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_diff1",
        "u_in_diff2",
        "u_out_lag1",
        "u_out_lag2",
        "u_out_diff1",
    ]

    print("\nCorrelation of Absolute Error (Inspiratory) with Features:")
    correlations = {}

    for i, name in enumerate(feature_names):
        feat_vals = inputs_flat[:, i]
        # Check for constant features to avoid division by zero in correlation
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(error_flat, feat_vals)[0, 1]
        correlations[name] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corr:
        print(f"{name}: {corr:.4f}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.19242813024255964

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        trainer = Trainer(model, device)
        predictions = trainer.predict(test_loader)

        # Load test metadata to map predictions to IDs
        test_meta = pd.read_csv(Config.TEST_META)

        # Ensure metadata is sorted by breath_id, id to match dataset order
        test_meta.sort_values(["breath_id", "id"], inplace=True)

        if len(predictions) != len(test_meta):
            print(
                f"Error: Prediction count {len(predictions)} does not match metadata {len(test_meta)}"
            )
        else:
            test_meta["pressure"] = predictions

            # Sort by 'id' for submission format
            test_meta.sort_values("id", inplace=True)

            submission_df = test_meta[["id", "pressure"]]
            submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
