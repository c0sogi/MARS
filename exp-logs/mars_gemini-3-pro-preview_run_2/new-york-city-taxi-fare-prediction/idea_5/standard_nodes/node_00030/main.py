import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import TaxiDataset
from library.trainer import Trainer


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for Fast Baseline requirements
    # We limit epochs to 1 to ensure completion within the time limit.
    Config.EPOCHS = 1

    # Ensure we use the full validation set for the metric.
    # We will manually subsample the training set later for speed.
    Config.DEBUG = False

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    print("Initializing Trainer and Model...")
    trainer = Trainer()

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading Datasets...")

    # Load Full Training Data
    # We use load_cached_data=True to utilize any existing processed data
    full_train_dataset = TaxiDataset(split="train", load_cached_data=True)

    # Subsample Training Data for Speed
    # 5 Million samples is sufficient for a strong baseline and fits within time limits
    TRAIN_SUBSET_SIZE = 5_000_000
    if len(full_train_dataset) > TRAIN_SUBSET_SIZE:
        # Use a fixed permutation for reproducibility based on seed
        indices = torch.randperm(len(full_train_dataset))[:TRAIN_SUBSET_SIZE]
        train_dataset = Subset(full_train_dataset, indices)
        print(f"Subsampled training set to {TRAIN_SUBSET_SIZE} samples.")
    else:
        train_dataset = full_train_dataset
        print(f"Using full training set ({len(train_dataset)} samples).")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Full Validation Data
    val_dataset = TaxiDataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    print("Starting Training...")

    # We manually execute the loop to control the flow and saving
    for epoch in range(Config.EPOCHS):
        # Train for one epoch
        train_loss = trainer.train_epoch(train_loader, epoch)

        # Validate
        val_rmse = trainer.validate(val_loader)
        print(f"Epoch {epoch+1} Validation RMSE: {val_rmse}")

        # Save Model (Always save the latest for this baseline)
        torch.save(trainer.model.state_dict(), Config.MODEL_SAVE_PATH)
        trainer.best_rmse = val_rmse

    # ---------------------------------------------------------
    # 4. Final Evaluation
    # ---------------------------------------------------------
    # Compute metric on the entire hold-out validation set
    final_rmse = trainer.validate(val_loader)
    print(f"Final Validation Metric: {final_rmse}")

    # ---------------------------------------------------------
    # 5. Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Failure Analysis ===")
    trainer.model.eval()

    all_errors = []
    all_features = []

    # Analyze a representative subset of validation data to save time/memory
    # 100 batches * 4096 ~ 400k samples, which is statistically significant
    ANALYSIS_BATCHES = 100

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= ANALYSIS_BATCHES:
                break

            cont_feat = batch["continuous_features"].to(trainer.device)
            cat_idx = batch["spatial_indices"].to(trainer.device)
            targets = batch["target"].to(trainer.device).view(-1, 1)

            outputs = trainer.model(cont_feat, cat_idx)
            preds = torch.clamp(outputs, min=Config.MIN_FARE_PREDICTION)

            # Calculate Absolute Error
            errors = torch.abs(preds - targets).cpu().numpy()
            feats = cont_feat.cpu().numpy()

            all_errors.append(errors)
            all_features.append(feats)

    if all_errors:
        flat_errors = np.concatenate(all_errors, axis=0).flatten()
        flat_features = np.concatenate(all_features, axis=0)

        print("Correlation between Error Magnitude and Input Features:")
        feature_names = val_dataset.continuous_cols

        for idx, col_name in enumerate(feature_names):
            if idx < flat_features.shape[1]:
                feature_vals = flat_features[:, idx]
                # Handle potential constant columns to avoid NaN correlation
                if np.std(feature_vals) > 0 and np.std(flat_errors) > 0:
                    corr = np.corrcoef(flat_errors, feature_vals)[0, 1]
                    print(f"  {col_name}: {corr:.4f}")
                else:
                    print(f"  {col_name}: NaN (Constant values)")
    else:
        print("No validation data available for analysis.")

    # ---------------------------------------------------------
    # 6. Submission Generation
    # ---------------------------------------------------------
    THRESHOLD = 4.278504866347902

    if final_rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({final_rmse}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nValidation RMSE ({final_rmse}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
