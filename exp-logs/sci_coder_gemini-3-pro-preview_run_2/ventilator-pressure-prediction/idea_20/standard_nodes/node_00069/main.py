import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import load_data
from library.model import CWCDP_BiLSTM
from library.loss import WeightedL1Loss
from library.utils import seed_everything
from library.train import train_epoch, validate
from library.inference import generate_predictions


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Patch Config for Fast Baseline Execution
    # We limit the training data and epochs to ensure quick runtime while maintaining performance.
    Config.DEBUG_SAMPLE_SIZE = 40000  # Use ~75% of training data (Limit max samples)
    Config.EPOCHS = 50  # Reduced from 200 for speed
    Config.T_MAX = 50  # Match epochs for scheduler

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")
    print(
        f"Training Config: {Config.EPOCHS} epochs, {Config.DEBUG_SAMPLE_SIZE} samples"
    )

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n--- Data Loading ---")
    # Train on a subset (debug=True with patched sample size)
    # We force reload_cached_data=False to ensure the new sample size is applied if a cache exists
    train_dataset = load_data("train", debug=True, load_cached_data=False)

    # Validate on the FULL hold-out set (debug=False) to get the correct metric
    val_dataset = load_data("val", debug=False, load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = CWCDP_BiLSTM().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = WeightedL1Loss(
        inspiratory_weight=Config.LOSS_INSPIRATORY_WEIGHT,
        expiratory_weight=Config.LOSS_EXPIRATORY_WEIGHT,
    )

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print("\n--- Starting Training ---")
    best_mae = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Logging
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val MAE: {val_mae:.5f}"
        )

        # Checkpoint
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_PATH)

    print(f"\nTraining Complete. Best Validation MAE: {best_mae}")

    # ==========================================
    # 5. Final Evaluation & Failure Analysis
    # ==========================================
    print("\n--- Final Evaluation ---")
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Compute Final Metric on Full Validation Set
    _, final_mae = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_mae}")

    # Failure Analysis: Correlation of Error with Features
    print("\n--- Failure Analysis ---")
    all_errors = []
    all_features = []

    # Collect data for analysis
    # We iterate through val_loader again to extract features and errors
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(x)
            abs_error = torch.abs(preds - y)

            # Filter for inspiratory phase (u_out < 0.5)
            mask = u_out < 0.5

            # We only care about errors in the inspiratory phase for the metric
            if mask.sum() > 0:
                # Flatten and filter
                batch_errors = abs_error[mask].cpu().numpy()
                batch_feats = x[mask].cpu().numpy()

                all_errors.append(batch_errors)
                all_features.append(batch_feats)

    if all_errors:
        all_errors = np.concatenate(all_errors)
        all_features = np.concatenate(all_features, axis=0)

        print("Correlation between Absolute Error and Input Features:")
        feature_names = Config.ALL_FEATURES

        for i, feat_name in enumerate(feature_names):
            feat_values = all_features[:, i]
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(all_errors, feat_values)[0, 1]
                print(f"  {feat_name}: {corr:.4f}")
            else:
                print(f"  {feat_name}: NaN (Constant)")
    else:
        print("No inspiratory phase data found for analysis.")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.1619843989610672

    if final_mae < THRESHOLD:
        print(
            f"\nMetric {final_mae} meets threshold ({THRESHOLD}). Generating submission..."
        )
        # Generate predictions on the full test set (debug=False)
        generate_predictions(debug=False, load_cached_data=True)
    else:
        print(
            f"\nMetric {final_mae} does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
