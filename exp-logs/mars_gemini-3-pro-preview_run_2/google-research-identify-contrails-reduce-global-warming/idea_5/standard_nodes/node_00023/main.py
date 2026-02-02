import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.loss import HybridBatchDiceLoss
from library.model import ContextEnhancedUNet
from library.dataset import ContrailsDataset
from library.training import train_one_epoch, validate
from library.inference import predict_and_submit

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Limit training to ensure completion within 2 hours
FAST_EPOCHS = 10
FAST_TRAIN_SAMPLES = 12000
SUBMISSION_THRESHOLD = 0.5454606988733747


def perform_failure_analysis(model, loader, metadata_df, device):
    """
    Computes per-sample error and correlates it with metadata features.
    """
    print("\nRunning Failure Analysis on Validation Set...")
    model.eval()
    results = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            r_ids = batch["record_id"]

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > Config.THRESHOLD).float()

            # Compute per-sample Dice
            # Flatten spatial dims: (B, 1, H, W) -> (B, H*W)
            preds_flat = preds.view(preds.size(0), -1)
            masks_flat = masks.view(masks.size(0), -1)

            intersection = (preds_flat * masks_flat).sum(dim=1)
            union = preds_flat.sum(dim=1) + masks_flat.sum(dim=1)

            # Dice = 2*Inter / Union. If Union is 0, Dice is 1.0
            dice_scores = (2.0 * intersection) / (union + 1e-6)
            dice_scores[union == 0] = 1.0

            for i, r_id in enumerate(r_ids):
                results.append({"record_id": str(r_id), "dice": dice_scores[i].item()})

    # Create DataFrame
    res_df = pd.DataFrame(results)

    # Merge with metadata
    # Ensure record_id is string for merging
    metadata_df = metadata_df.copy()
    metadata_df["record_id"] = metadata_df["record_id"].astype(str)

    merged = res_df.merge(metadata_df, on="record_id", how="left")
    merged["error"] = 1.0 - merged["dice"]

    # Calculate Correlations
    features = ["timestamp", "row_min", "col_min"]
    print("Correlation between Error Magnitude (1-Dice) and Input Features:")

    for feat in features:
        if feat in merged.columns:
            corr = merged["error"].corr(merged[feat])
            print(f"  {feat}: {corr:.6f}")
        else:
            print(f"  {feat}: Not found in metadata")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Load and Prepare Data
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)

    # Create Datasets
    train_dataset = ContrailsDataset(train_df, train=True)
    val_dataset = ContrailsDataset(val_df, train=True)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model and Training Components
    model = ContextEnhancedUNet().to(device)

    criterion = HybridBatchDiceLoss(
        bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=FAST_EPOCHS, eta_min=1e-6)

    # 4. Training Loop
    print(f"Starting training for {FAST_EPOCHS} epochs...")
    best_dice = -1.0

    for epoch in range(FAST_EPOCHS):
        start_t = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        # Save Best Model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

        print(
            f"Epoch {epoch+1}/{FAST_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.6f} | "
            f"Time: {time.time() - start_t:.1f}s"
        )

    print("Training complete.")

    # 5. Final Validation and Failure Analysis
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Compute Final Metric on Full Validation Set
    _, final_metric = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    perform_failure_analysis(model, val_loader, val_df, device)

    # 6. Submission
    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        predict_and_submit(
            model_path=Config.BEST_MODEL_PATH,
            metadata_path=Config.TEST_METADATA_PATH,
            output_path=Config.SUBMISSION_PATH,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            device=device,
            debug=False,
        )
    else:
        print(
            f"\nMetric ({final_metric}) did not exceed threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
