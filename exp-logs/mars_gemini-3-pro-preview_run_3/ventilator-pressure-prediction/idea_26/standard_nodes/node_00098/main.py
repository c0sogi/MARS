import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

# Import from provided library
from library.config import Config
from library.model import DKRHNet
from library.train_utils import (
    MaskedL1Loss,
    train_epoch,
    validate_epoch,
    generate_submission,
)
from library.data_utils import get_transformed_data


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for Fast Baseline
    # 80 epochs is too long for a 2-hour limit with full data.
    # 6 epochs should take ~45 mins on A100 and provide a good baseline.
    Config.EPOCHS = 6

    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading and processing data...")
    # This handles caching, feature engineering, scaling, and reshaping
    train_loader, val_loader, test_loader = get_transformed_data(load_cached_data=True)

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("Initializing DKRH-Net...")
    model = DKRHNet().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    # Loss Function
    criterion = MaskedL1Loss()

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss = validate_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val Loss (Proxy): {best_val_loss:.6f}")

    # ---------------------------------------------------------
    # 5. Final Validation & Failure Analysis
    # ---------------------------------------------------------
    print("Performing final validation and failure analysis...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Generate predictions on Validation Set
    val_preds = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            preds = model(inputs)
            val_preds.append(preds.cpu().numpy())

    val_preds_flat = np.concatenate(val_preds).flatten()

    # Load Raw Validation Data for Accurate Metric Calculation
    # We do this to avoid issues with scaled u_out in the loader
    val_df_raw = pd.read_csv(Config.VAL_PATH)
    y_true = val_df_raw["pressure"].values
    u_out_true = val_df_raw["u_out"].values

    # Ensure lengths match
    if len(y_true) != len(val_preds_flat):
        print(
            f"Warning: Length mismatch. True: {len(y_true)}, Preds: {len(val_preds_flat)}"
        )
        # Truncate to safe length (though this shouldn't happen)
        min_len = min(len(y_true), len(val_preds_flat))
        y_true = y_true[:min_len]
        u_out_true = u_out_true[:min_len]
        val_preds_flat = val_preds_flat[:min_len]
        val_df_raw = val_df_raw.iloc[:min_len]

    # Calculate Metric (MAE on Inspiratory Phase)
    insp_mask = u_out_true == 0
    final_metric = np.mean(np.abs(y_true[insp_mask] - val_preds_flat[insp_mask]))

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    # Calculate absolute error
    val_df_raw["abs_error"] = np.abs(y_true - val_preds_flat)

    # Filter for inspiratory phase for analysis
    insp_df = val_df_raw[val_df_raw["u_out"] == 0]

    # Calculate correlations
    # We check correlation of error with key features
    corr_cols = ["R", "C", "u_in", "time_step"]
    correlations = (
        insp_df[corr_cols + ["abs_error"]].corr()["abs_error"].drop("abs_error")
    )

    print("Error Correlations (Inspiratory Phase):")
    print(correlations)

    # ---------------------------------------------------------
    # 6. Submission
    # ---------------------------------------------------------
    threshold = 0.16391726930343686

    if final_metric < threshold:
        print(f"Metric {final_metric:.6f} < {threshold}. Generating submission...")
        generate_submission(test_loader, device, best_model_path)
    else:
        print(f"Metric {final_metric:.6f} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
