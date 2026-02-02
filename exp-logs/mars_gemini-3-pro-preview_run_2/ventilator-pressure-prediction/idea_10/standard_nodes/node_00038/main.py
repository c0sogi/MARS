import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, WeightedL1Loss
from library.dataset import load_and_preprocess_data
from library.model import RGIBiLSTM
from library.train import train_one_epoch, validate_one_epoch
from library.inference import predict


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline Execution
    Config.EPOCHS = 20  # Reduced to ensure completion within time limit
    Config.BATCH_SIZE = 256  # Standard batch size for stability
    Config.NUM_WORKERS = 12  # Utilize available vCPUs

    # Setup Device and Seed
    device = torch.device(Config.DEVICE)
    seed_everything(Config.SEED)
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading and preprocessing data...")
    # load_cached_data=True will use existing parquet files if available
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data(
        load_cached_data=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing RGI-BiLSTM model...")
    model = RGIBiLSTM().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.SCHEDULER_MIN_LR
    )

    criterion = WeightedL1Loss()

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_val_mae = float("inf")
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, Config.MAX_GRAD_NORM
        )

        # Validate
        val_mae = validate_one_epoch(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.6f} | Time: {elapsed:.2f}s"
        )

        # Save Best Model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
            print(f"  -> Model saved (Improved MAE: {best_val_mae:.6f})")

    print("Training complete.")

    # ==========================================
    # 5. Failure Analysis & Final Metric
    # ==========================================
    print("\nRunning Failure Analysis on Validation Set...")

    # Load best model
    if os.path.exists(Config.MODEL_CHECKPOINT):
        model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT))
    else:
        print("Warning: No checkpoint found. Using current model state.")

    model.eval()

    val_preds_list = []
    val_targets_list = []
    val_inputs_list = []
    val_u_out_list = []

    # Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            X = batch["X"].to(device)
            u_out = batch["u_out"].to(device)
            y = batch["y"].to(device)

            preds = model(X)

            val_preds_list.append(preds.cpu())
            val_targets_list.append(y.cpu())
            val_inputs_list.append(X.cpu())
            val_u_out_list.append(u_out.cpu())

    # Concatenate all batches
    val_preds = torch.cat(val_preds_list)
    val_targets = torch.cat(val_targets_list)
    val_inputs = torch.cat(val_inputs_list)
    val_u_out = torch.cat(val_u_out_list)

    # Calculate Absolute Errors
    abs_errors = torch.abs(val_preds - val_targets)

    # Calculate Final Metric (Inspiratory Phase Only: u_out == 0)
    insp_mask = val_u_out == 0
    final_metric = abs_errors[insp_mask].mean().item()

    # PRINT FINAL METRIC
    print(f"Final Validation Metric: {final_metric}")

    # Correlation Analysis
    # Flatten to (N_samples * Seq_len, Features)
    flat_inputs = val_inputs.reshape(-1, val_inputs.shape[-1]).numpy()
    flat_errors = abs_errors.flatten().numpy()
    flat_u_out = val_u_out.flatten().numpy()

    # Filter for Inspiratory Phase for analysis (most relevant to metric)
    insp_indices = flat_u_out == 0

    analysis_df = pd.DataFrame(flat_inputs[insp_indices], columns=Config.FEATURE_COLS)
    analysis_df["error"] = flat_errors[insp_indices]

    print(
        "\nCorrelation between Input Features and Error Magnitude (Inspiratory Phase):"
    )
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False)
    )
    print(correlations)

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.19242813024255964

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        # Use library inference function
        predict(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS)
    else:
        print(
            f"\nMetric {final_metric} does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
