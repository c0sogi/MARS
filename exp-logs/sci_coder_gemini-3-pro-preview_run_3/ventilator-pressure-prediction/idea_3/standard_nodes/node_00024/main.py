import os
import sys
import torch
import numpy as np
import pandas as pd
import glob

# Import from provided library files
from library.config import Config
from library.model import ParallelTCNLSTM
from library.data_loader import get_dataloaders
from library.train_eval import train_epoch, validate, generate_submission_file, set_seed


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("Initializing Runfile...")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Ensure fresh feature engineering
    Config.FORCE_RECOMPUTE = True

    # Clean up existing cache to enforce pipeline integrity
    if Config.FORCE_RECOMPUTE:
        print("Clearing cache...")
        cache_files = glob.glob(os.path.join(Config.CACHE_DIR, "*.npy"))
        for f in cache_files:
            try:
                os.remove(f)
            except OSError:
                pass

    device = torch.device(Config.HYPERPARAMS["device"])
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading data...")
    # get_dataloaders handles feature engineering, scaling, and caching
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.HYPERPARAMS["batch_size"],
        num_workers=Config.HYPERPARAMS["num_workers"],
        load_cached_data=not Config.FORCE_RECOMPUTE,
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing model...")
    model = ParallelTCNLSTM(Config.HYPERPARAMS).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.HYPERPARAMS["learning_rate"],
        weight_decay=Config.HYPERPARAMS["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.HYPERPARAMS["scheduler_factor"],
        patience=Config.HYPERPARAMS["scheduler_patience"],
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("Starting training...")
    best_val_mae = float("inf")
    early_stop_counter = 0
    save_path = Config.MODEL_SAVE_PATH

    for epoch in range(Config.HYPERPARAMS["epochs"]):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.HYPERPARAMS['epochs']} | Train MAE: {train_loss:.6f} | Val MAE: {val_loss:.6f}"
        )

        # Save Best Model
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= Config.HYPERPARAMS["patience"]:
            print("Early stopping triggered.")
            break

    print("Training finished.")

    # -------------------------------------------------------------------------
    # 5. Final Validation and Failure Analysis
    # -------------------------------------------------------------------------
    print("Performing Failure Analysis on Best Model...")

    # Load best model
    model.load_state_dict(torch.load(save_path))
    model.eval()

    val_preds_list = []
    val_targets_list = []
    val_inputs_list = []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            preds = model(X)

            val_preds_list.append(preds.cpu().numpy())
            val_targets_list.append(y.cpu().numpy())
            val_inputs_list.append(X.cpu().numpy())

    # Concatenate all batches
    val_preds = np.concatenate(val_preds_list, axis=0)  # (N, 80, 1)
    val_targets = np.concatenate(val_targets_list, axis=0)  # (N, 80, 1)
    val_inputs = np.concatenate(val_inputs_list, axis=0)  # (N, 80, F)

    # Flatten for analysis
    N, L, F = val_inputs.shape
    val_preds_flat = val_preds.reshape(-1)
    val_targets_flat = val_targets.reshape(-1)
    val_inputs_flat = val_inputs.reshape(-1, F)

    # Extract u_out for masking (Index 2 based on Config.FEATURE_COLS)
    # ["time_step", "u_in", "u_out", "R", "C", ...]
    # Note: Inputs are scaled. u_out mean is ~0.6.
    # Inspiratory (u_out=0) -> Scaled value < 0
    # Expiratory (u_out=1) -> Scaled value > 0
    u_out_scaled = val_inputs_flat[:, 2]
    inspiratory_mask = u_out_scaled < 0

    # Calculate MAE on Inspiratory Phase
    errors = np.abs(val_preds_flat - val_targets_flat)
    final_metric = np.mean(errors[inspiratory_mask])

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation of Error with Features (Inspiratory Phase only)
    print("\nFailure Analysis (Correlation of Absolute Error with Features):")

    # Filter data to inspiratory phase
    error_insp = errors[inspiratory_mask]
    inputs_insp = val_inputs_flat[inspiratory_mask]

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(inputs_insp, columns=Config.FEATURE_COLS)
    analysis_df["abs_error"] = error_insp

    # Calculate correlations
    correlations = analysis_df.corr()["abs_error"].sort_values(ascending=False)
    print(correlations)

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.23978149890899658

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} meets threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission_file(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric {final_metric} does NOT meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
