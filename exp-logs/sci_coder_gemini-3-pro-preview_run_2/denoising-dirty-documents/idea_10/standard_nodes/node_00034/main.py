import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

from library.utils import seed_everything, get_device, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders
from library.model import WaveCACResUNet, train_one_epoch, validate
from library.inference import generate_submission


def main():
    # --- Configuration ---
    DATA_DIR = "./input"
    WORK_DIR = "./working/idea_10"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_PATH = os.path.join(WORK_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Hyperparameters for Fast Baseline
    EPOCHS = 20
    BATCH_SIZE = 32
    LR = 1e-3
    WEIGHT_DECAY = 1e-2
    # High-density sampling: 50 patches per image per epoch (92 * 50 = 4600 samples/epoch)
    TRAIN_SAMPLES_PER_EPOCH = 50
    PATIENCE = 5
    SUBMISSION_THRESHOLD = 0.0076658159

    # --- Setup ---
    seed_everything(42)
    device = get_device()
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    print(f"Device: {device}")
    print("Initializing DataLoaders...")

    # Get DataLoaders
    # We use cached loading and high-density sampling for training
    train_loader, val_loader = get_dataloaders(
        data_dir=DATA_DIR,
        cache_dir=CACHE_DIR,
        batch_size=BATCH_SIZE,
        num_workers=2,
        patch_size=128,
        train_samples_per_epoch=TRAIN_SAMPLES_PER_EPOCH,
        val_samples_per_epoch=1,
    )

    # --- Model Initialization ---
    print("Initializing Model...")
    model = WaveCACResUNet(in_channels=1, base_filters=64).to(device)

    # Optimization
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- Training Loop ---
    print("Starting Training...")
    best_val_metric = float("inf")
    no_improve_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate (Monitoring metric)
        val_rmse_monitor = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Checkpointing
        if val_rmse_monitor < best_val_metric:
            best_val_metric = val_rmse_monitor
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_rmse": best_val_metric,
                    "optimizer": optimizer.state_dict(),
                },
                CHECKPOINT_PATH,
            )
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early Stopping
        if no_improve_epochs >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # --- Final Evaluation & Failure Analysis ---
    print("Loading best model for evaluation...")
    load_checkpoint(CHECKPOINT_PATH, model)
    model.eval()

    print("Calculating Final Validation Metric and performing Failure Analysis...")

    total_squared_error = 0.0
    total_pixels = 0

    # Containers for correlation analysis
    # We'll sample pixels if the dataset is too large, but for 23 val images it's manageable
    all_errors = []
    all_inputs = []

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Predict Noise
            pred_noise = model(noisy)

            # Reconstruct Clean Image
            pred_clean = noisy - pred_noise
            pred_clean = torch.clamp(pred_clean, 0, 1)

            # Calculate Errors
            diff = clean - pred_clean
            squared_diff = diff**2
            abs_diff = torch.abs(diff)

            # Accumulate Global RMSE stats
            total_squared_error += torch.sum(squared_diff).item()
            total_pixels += clean.numel()

            # Collect data for failure analysis (flattened)
            # Move to CPU to save GPU memory
            all_errors.append(abs_diff.cpu().numpy().flatten())
            all_inputs.append(noisy.cpu().numpy().flatten())

    # 1. Final Validation Metric (Global RMSE)
    final_global_rmse = np.sqrt(total_squared_error / total_pixels)
    print(f"Final Validation Metric: {final_global_rmse}")

    # 2. Failure Analysis
    # Concatenate all collected pixels
    flat_errors = np.concatenate(all_errors)
    flat_inputs = np.concatenate(all_inputs)

    # Calculate Pearson Correlation between Error Magnitude and Input Intensity
    # This tells us if the model struggles more with bright (background) or dark (text) areas
    corr, _ = pearsonr(flat_errors, flat_inputs)
    print(f"Correlation between Error Magnitude and Input Intensity: {corr}")

    # --- Conditional Submission ---
    if final_global_rmse < SUBMISSION_THRESHOLD:
        print(
            f"Metric {final_global_rmse} meets threshold {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        generate_submission(
            checkpoint_path=CHECKPOINT_PATH,
            data_dir=DATA_DIR,
            output_path=SUBMISSION_PATH,
            cache_dir=CACHE_DIR,
            device_name=device.type,
        )
    else:
        print(
            f"Metric {final_global_rmse} does not meet threshold {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
