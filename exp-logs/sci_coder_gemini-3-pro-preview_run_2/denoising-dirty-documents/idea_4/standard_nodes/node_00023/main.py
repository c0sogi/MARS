import os
import time
import torch
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.model import ResUNet
from library.dataset import DenoisingDataset
from library.train import (
    train_one_epoch,
    validate,
    DeepSupervisionLoss,
    pad_to_multiple,
    unpad,
)
from library.inference import generate_submission


def run():
    # -------------------------------------------------------------------------
    # 1. Setup and Initialization
    # -------------------------------------------------------------------------
    Config.initialize()
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    # Load cached data is True by default in dataset class
    train_dataset = DenoisingDataset(mode="train")
    val_dataset = DenoisingDataset(mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Validation loader must have batch_size=1 for accurate full-image evaluation
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Model, Optimizer, Scheduler
    # -------------------------------------------------------------------------
    model = ResUNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = DeepSupervisionLoss()

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    best_rmse = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_rmse = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Time: {elapsed:.1f}s | "
            f"LR: {current_lr:.2e} | Train Loss: {train_loss:.5f} | Val RMSE: {val_rmse:.6f}"
        )

        # Checkpointing
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            save_checkpoint(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # -------------------------------------------------------------------------
    # 5. Final Validation Metric
    # -------------------------------------------------------------------------
    print(f"Final Validation Metric: {best_rmse}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis on Best Model...")

    # Reload best model
    load_checkpoint(model, Config.MODEL_SAVE_PATH, device=device)
    model.eval()

    img_rmses = []
    img_means = []
    img_stds = []

    with torch.no_grad():
        for noisy, target in val_loader:
            noisy = noisy.to(device)
            target = target.to(device)

            # Pad for inference
            noisy_padded, h, w = pad_to_multiple(noisy, 16)

            # Predict
            pred_residual_padded = model(noisy_padded)

            # Unpad
            pred_residual = unpad(pred_residual_padded, h, w)

            # Reconstruct Clean Images for Metric Calculation
            # Clean = Noisy - Residual
            gt_clean = torch.clamp(noisy - target, 0, 1)
            pred_clean = torch.clamp(noisy - pred_residual, 0, 1)

            # Calculate RMSE for this specific image
            diff = (pred_clean - gt_clean).cpu().numpy().flatten()
            img_rmse = np.sqrt(np.mean(diff**2))
            img_rmses.append(img_rmse)

            # Calculate Input Features (Noisy Image Stats)
            noisy_np = noisy.cpu().numpy().flatten()
            img_means.append(np.mean(noisy_np))
            img_stds.append(np.std(noisy_np))

    # Calculate Correlations
    if len(img_rmses) > 1:
        corr_mean, _ = pearsonr(img_rmses, img_means)
        corr_std, _ = pearsonr(img_rmses, img_stds)

        print(f"Correlation (Error vs Input Mean Intensity): {corr_mean:.4f}")
        print(f"Correlation (Error vs Input Std Dev): {corr_std:.4f}")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    SUBMISSION_THRESHOLD = 0.009138691164531186

    if best_rmse < SUBMISSION_THRESHOLD:
        print(
            f"\nValidation RMSE ({best_rmse}) is below threshold ({SUBMISSION_THRESHOLD})."
        )
        generate_submission(output_path=Config.SUBMISSION_FILE_PATH)
    else:
        print(
            f"\nValidation RMSE ({best_rmse}) did not meet threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
