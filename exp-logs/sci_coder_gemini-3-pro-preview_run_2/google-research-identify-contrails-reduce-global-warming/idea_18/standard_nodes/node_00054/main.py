import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.dataset import ContrailDataset
from library.network import AttentionGatedUNet
from library.engine import train_model, validate
from library.inference import predict_and_submit
from library.utils import dice_coef_metric


def perform_failure_analysis(model, val_loader, val_dataset, device):
    """
    Analyzes model performance on the validation set to identify systematic errors.
    Calculates per-sample Dice scores and correlates error (1 - Dice) with metadata.
    """
    print("\nStarting Failure Analysis...")
    model.eval()

    # Store per-sample errors
    errors = []

    # Ensure loader is not shuffled to match dataset dataframe order
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > Config.THRESHOLD).float()

            # Compute per-sample Dice (Error = 1 - Dice)
            # Batch shape: (B, 1, H, W)
            for i in range(images.size(0)):
                pred_flat = preds[i].view(-1)
                mask_flat = masks[i].view(-1)

                intersection = (pred_flat * mask_flat).sum().item()
                union = pred_flat.sum().item() + mask_flat.sum().item()

                epsilon = 1e-6
                dice = (2.0 * intersection) / (union + epsilon)
                errors.append(1.0 - dice)

    # Add errors to the validation dataframe copy
    df_analysis = val_dataset.df.copy()

    # Truncate df if loader didn't cover everything (shouldn't happen if configured correctly)
    if len(errors) != len(df_analysis):
        print(
            f"Warning: Mismatch in analysis lengths. Errors: {len(errors)}, DF: {len(df_analysis)}"
        )
        df_analysis = df_analysis.iloc[: len(errors)]

    df_analysis["error"] = errors

    # Extract numerical features for correlation
    # timestamp, row_min (lat proxy), col_min (lon proxy)
    features = ["timestamp", "row_min", "col_min", "row_size", "col_size"]
    correlations = {}

    print("Correlation between Error Magnitude (1-Dice) and Metadata features:")
    for feat in features:
        if feat in df_analysis.columns:
            corr = df_analysis[feat].corr(df_analysis["error"])
            correlations[feat] = corr
            print(f"  {feat}: {corr:.4f}")

    return correlations


def main():
    # 1. Setup
    Config.set_seed(Config.SEED)
    device = Config.DEVICE

    # Modify Config for Fast Baseline Execution
    # We limit training data and epochs to ensure completion within 2 hours
    # while retaining enough capacity to learn.
    Config.MAX_TRAIN_SAMPLES = 5000  # Train on a subset
    Config.EPOCHS = 8  # Reduce epochs

    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Training Samples: {Config.MAX_TRAIN_SAMPLES}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")

    # 2. Data Loading
    print("Loading Datasets...")
    # Train on subset
    train_dataset = ContrailDataset(split="train", max_samples=Config.MAX_TRAIN_SAMPLES)
    # Validate on FULL set to get comparable metric
    val_dataset = ContrailDataset(split="validation", max_samples=None)

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
        shuffle=False,  # Important for failure analysis mapping
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = AttentionGatedUNet(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # 4. Training
    print("Starting Training...")
    best_dice = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=Config.EPOCHS,
        patience=5,
    )

    # 5. Final Validation
    print("Performing Final Validation...")
    # Load best model weights
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_metric = validate(model, val_loader, device, threshold=Config.THRESHOLD)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, val_dataset, device)

    # 7. Submission
    # Threshold condition from task description
    SUBMISSION_THRESHOLD = 0.5910660985501295

    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) > Threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        # Free up memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

        # Run inference
        predict_and_submit(
            checkpoint_path=best_model_path,
            batch_size=Config.BATCH_SIZE * 2,
            device=device,
            max_samples=None,  # Predict on full test set
        )
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
