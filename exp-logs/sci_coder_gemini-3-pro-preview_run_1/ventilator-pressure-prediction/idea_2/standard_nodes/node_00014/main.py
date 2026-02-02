import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import sys

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.data_processing import DataProcessor
from library.dataset import VentilatorDataset
from library.model import HybridCNNLSTM
from library.training import train_epoch, validate_epoch, MaskedL1Loss
from library.inference import generate_predictions


def main():
    # 1. Configuration & Setup
    # Override epochs for a fast baseline execution
    Config.EPOCHS = 15

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Preparation
    processor = DataProcessor()

    # Load Training Data
    print("Loading training data...")
    X_train, y_train, u_out_train = processor.load_dataset(
        split="train", load_cached_data=True
    )
    train_dataset = VentilatorDataset(X_train, u_out_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Load Validation Data
    print("Loading validation data...")
    X_val, y_val, u_out_val = processor.load_dataset(split="val", load_cached_data=True)
    val_dataset = VentilatorDataset(X_val, u_out_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = HybridCNNLSTM().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = MaskedL1Loss()

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
        anneal_strategy="cos",
    )

    # 4. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_val_loss = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, epoch)

        # Checkpoint
        if val_loss < best_val_loss:
            print(
                f"New best model found! Loss: {val_loss:.6f} -> Saving to {Config.MODEL_PATH}"
            )
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)

    print("Training complete.")

    # 5. Final Assessment & Failure Analysis
    print("\n=== Final Assessment ===")

    # Load best model
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Warning: No model file found. Using current model state.")

    model.eval()

    # Generate predictions on validation set for analysis
    val_preds = []
    with torch.no_grad():
        for X, _, _ in val_loader:
            X = X.to(device)
            preds = model(X)
            val_preds.append(preds.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)

    # Calculate Final Validation Metric (MAE on inspiratory phase)
    # Mask: 1 for inspiratory (u_out=0), 0 for expiratory
    mask = 1.0 - u_out_val
    abs_error = np.abs(val_preds - y_val)
    masked_error = abs_error * mask

    final_metric = np.sum(masked_error) / (np.sum(mask) + 1e-8)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n=== Failure Analysis ===")
    # Flatten arrays for correlation analysis
    # Only consider inspiratory phase for failure analysis as that's what we are scored on
    valid_indices = mask.flatten() == 1.0

    flat_error = abs_error.flatten()[valid_indices]
    flat_X = X_val.reshape(-1, X_val.shape[-1])[valid_indices]

    # Reconstruct feature names based on Config
    # Continuous
    feature_names = Config.CONT_FEATURES.copy()
    # One-Hot R
    feature_names.extend([f"R_{v}" for v in Config.R_VALUES])
    # One-Hot C
    feature_names.extend([f"C_{v}" for v in Config.C_VALUES])
    # u_out
    feature_names.append("u_out")

    # Calculate correlations
    print("Correlation between Absolute Error and Features (Inspiratory Phase):")
    correlations = {}
    for i, name in enumerate(feature_names):
        # Calculate Pearson correlation
        feat_values = flat_X[:, i]
        # Avoid constant features causing division by zero (e.g. u_out is all 0 in inspiratory phase)
        if np.std(feat_values) > 1e-9:
            corr = np.corrcoef(flat_error, feat_values)[0, 1]
            correlations[name] = corr
        else:
            correlations[name] = 0.0

    # Sort and print top correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corr[:5]:
        print(f"  {name}: {corr:.4f}")

    # 6. Submission Generation
    # Threshold check
    if final_metric < 0.5275861736753809:
        print(f"\nMetric {final_metric} < 0.5275861736753809. Generating submission...")
        generate_predictions(
            model_path=Config.MODEL_PATH,
            output_path=Config.SUBMISSION_PATH,
            sample_submission_path=Config.SAMPLE_SUBMISSION_PATH,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            device=Config.DEVICE,
            load_cached_data=True,
        )
    else:
        print(
            f"\nMetric {final_metric} >= 0.5275861736753809. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
