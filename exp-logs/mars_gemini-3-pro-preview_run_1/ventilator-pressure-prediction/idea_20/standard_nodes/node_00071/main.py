import sys
import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Ensure library can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, compute_metric
from library.dataset import prepare_datasets
from library.model import CuratedIdentityNet
from library.loss import MaskedL1Loss
from library.train import train_one_epoch, validate, generate_submission


def main():
    # --- Configuration ---
    # Override epochs for fast baseline execution within time limits
    # 15 epochs is sufficient for convergence with OneCycleLR on this dataset
    Config.EPOCHS = 15

    # Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # --- Data Preparation ---
    print("Preparing datasets...")
    # load_cached_data=True to utilize any existing preprocessed files
    train_loader, val_loader, test_loader, test_ids = prepare_datasets(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # --- Model Initialization ---
    print("Initializing model...")
    model = CuratedIdentityNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    total_steps = Config.EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = MaskedL1Loss()

    # --- Training Loop ---
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_val_mae = float("inf")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            Config.AUX_LOSS_WEIGHT,
        )

        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.6f}"
        )

        # Save Best Model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best model saved! MAE: {best_val_mae:.6f}")

    print("Training complete.")

    # --- Final Evaluation & Failure Analysis ---
    print("\nRunning Final Evaluation and Failure Analysis...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Collect all validation predictions and metadata
    all_preds = []
    all_targets = []
    all_u_outs = []

    # For correlation analysis
    meta_mae = []
    meta_R = []
    meta_C = []
    meta_u_in_mean = []

    # Feature indices
    # u_in is in Config.CONT_FEATURES.
    # Config.CONT_FEATURES = ["time_step", "u_in", "R", "C", ...]
    try:
        u_in_idx = Config.CONT_FEATURES.index("u_in")
    except ValueError:
        u_in_idx = 1  # Fallback, usually index 1 after time_step

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            static = batch["static"].to(device)
            u_out = batch["u_out"].to(device)
            y = batch["y"].to(device)

            # Forward
            pred, _ = model(x, static)
            pred = pred.squeeze(-1)  # (B, 80)

            # --- Global Metric Accumulation ---
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_u_outs.append(u_out.cpu().numpy())

            # --- Failure Analysis (Per Breath) ---
            # x: (B, 80, F)
            # static: (B, 2) -> R, C

            x_np = x.cpu().numpy()
            static_np = static.cpu().numpy()
            u_out_np = u_out.cpu().numpy()
            y_np = y.cpu().numpy()
            pred_np = pred.cpu().numpy()

            # Iterate through breaths in batch
            for i in range(len(x_np)):
                # Mask for this breath (inspiratory phase)
                mask = u_out_np[i] == 0

                if mask.sum() > 0:
                    # MAE for this breath
                    error = np.abs(pred_np[i][mask] - y_np[i][mask])
                    mae = np.mean(error)

                    meta_mae.append(mae)
                    meta_R.append(static_np[i, 0])
                    meta_C.append(static_np[i, 1])
                    meta_u_in_mean.append(np.mean(x_np[i, :, u_in_idx]))

    # Compute Global Metric
    flat_preds = np.concatenate(all_preds).flatten()
    flat_targets = np.concatenate(all_targets).flatten()
    flat_u_outs = np.concatenate(all_u_outs).flatten()

    final_metric = compute_metric(flat_preds, flat_targets, flat_u_outs)
    print(f"Final Validation Metric: {final_metric}")

    # Compute Correlations
    df_analysis = pd.DataFrame(
        {"mae": meta_mae, "R": meta_R, "C": meta_C, "u_in_mean": meta_u_in_mean}
    )

    print("\nFailure Analysis (Correlation of Error with Features):")
    correlations = df_analysis.corr()["mae"].sort_values(ascending=False)
    print(correlations)

    # --- Submission ---
    threshold = 0.2164510190486908
    if final_metric < threshold:
        print(
            f"\nMetric passed threshold ({final_metric:.6f} < {threshold}). Generating submission..."
        )
        generate_submission(
            model, test_loader, test_ids, device, Config.SUBMISSION_PATH
        )
    else:
        print(
            f"\nMetric failed threshold ({final_metric:.6f} >= {threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
