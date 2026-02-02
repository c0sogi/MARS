import os
import sys
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.dataset import get_datasets
from library.model import SegFormer
from library.trainer import Trainer
from library.inference import z_scan_predict
from library.utils import seed_everything, fbeta_score


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # load_cached_data=True to use pre-computed .npy files if available
    train_ds, val_ds = get_datasets(load_cached_data=True)

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = SegFormer().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # 4. Training
    print("Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # Run training
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # 5. Final Validation & Failure Analysis
    print("Performing final validation and failure analysis...")

    # Load best model
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model found. Using current model state.")

    model.eval()

    all_preds = []
    all_targets = []

    # For failure analysis
    batch_errors = []
    batch_intensities = []

    with torch.no_grad():
        for images, labels, masks, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Collect for metric
            all_preds.append(probs.cpu())
            all_targets.append(labels.cpu())

            # Failure Analysis Data Collection
            # Calculate Mean Absolute Error per sample in batch
            # Shape: (B, 1, H, W) -> (B,)
            mae = torch.abs(probs - labels).mean(dim=(1, 2, 3))

            # Calculate Mean Intensity per sample in batch
            # Shape: (B, 3, H, W) -> (B,)
            intensity = images.mean(dim=(1, 2, 3))

            batch_errors.append(mae.cpu().numpy())
            batch_intensities.append(intensity.cpu().numpy())

    # Concatenate
    all_preds_t = torch.cat(all_preds, dim=0).view(-1)
    all_targets_t = torch.cat(all_targets, dim=0).view(-1)

    # Compute Final Metric
    final_f05 = fbeta_score(
        all_preds_t, all_targets_t, beta=Config.F_BETA, threshold=Config.MASK_THRESHOLD
    )

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_f05}")

    # Failure Analysis Calculation
    errors_flat = np.concatenate(batch_errors)
    intensities_flat = np.concatenate(batch_intensities)

    if len(errors_flat) > 1:
        corr, p_val = pearsonr(intensities_flat, errors_flat)
        print(
            f"Failure Analysis - Correlation (Input Intensity vs Error): {corr:.4f} (p={p_val:.4f})"
        )
        if abs(corr) > 0.3:
            print(
                "Observation: Significant correlation found. Model performance varies with ink intensity/contrast."
            )
        else:
            print(
                "Observation: No strong linear correlation between intensity and error."
            )

    # 6. Submission Logic
    # Threshold from prompt
    SUBMISSION_THRESHOLD = 0.597622633

    if final_f05 > SUBMISSION_THRESHOLD:
        print(
            f"Validation score ({final_f05:.6f}) exceeds baseline ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        # Run inference using the Decoupled Z-Scanning strategy implemented in library.inference
        z_scan_predict(load_cached_data=True)
    else:
        print(
            f"Validation score ({final_f05:.6f}) did not exceed baseline ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
