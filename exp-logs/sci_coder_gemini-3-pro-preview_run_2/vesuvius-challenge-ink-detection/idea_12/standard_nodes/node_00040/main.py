import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import InkDataset
from library.model import SegFormerB3
from library.training import BCEDiceLoss, train_one_epoch, validate
from library.inference import z_scan_inference


def main():
    # 1. Setup and Configuration
    Config.setup()
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Data Preparation
    # Using the full dataset as it is small (412 train samples) and fits within the "fast baseline" requirement.
    train_dataset = InkDataset(mode="train")
    val_dataset = InkDataset(mode="validation")

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

    # 3. Model Initialization
    model = SegFormerB3().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    criterion = BCEDiceLoss()

    # 4. Training Loop
    best_val_f05 = -1.0
    model_save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_f05, val_dice = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_f05)

        # Checkpoint
        if val_f05 > best_val_f05:
            best_val_f05 = val_f05
            torch.save(model.state_dict(), model_save_path)

    # Required Output: Final Validation Metric
    print(f"Final Validation Metric: {best_val_f05}")

    # 5. Failure Analysis
    # Load the best model for analysis
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    all_errors = []
    all_intensities = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Predict
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Calculate Mean Absolute Error per sample in the batch
            # Error shape: (Batch, 1, H, W) -> reduce to (Batch,)
            error_map = torch.abs(probs - masks)
            batch_errors = error_map.mean(dim=[1, 2, 3]).cpu().numpy()

            # Calculate Mean Intensity per sample (Input Feature)
            # Image shape: (Batch, 3, H, W) -> reduce to (Batch,)
            batch_intensities = images.mean(dim=[1, 2, 3]).cpu().numpy()

            all_errors.extend(batch_errors)
            all_intensities.extend(batch_intensities)

    # Compute Correlation
    if len(all_errors) > 1:
        correlation = np.corrcoef(all_errors, all_intensities)[0, 1]
        print(f"Correlation between Error Magnitude and Mean Intensity: {correlation}")
    else:
        print("Insufficient data for failure analysis.")

    # 6. Submission Generation
    # Condition: Generate submission only if metric > 0.597622633
    SUBMISSION_THRESHOLD = 0.597622633

    if best_val_f05 > SUBMISSION_THRESHOLD:
        z_scan_inference()
    else:
        print(
            f"Validation metric {best_val_f05} did not exceed threshold {SUBMISSION_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
