import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.dataset import ManufacturingDataset
from library.model import HybridResFunnel
from library.engine import Trainer, predict


def main():
    # --------------------------------------------------------------------------
    # 1. Setup
    # --------------------------------------------------------------------------
    seed_everything(Config.RANDOM_SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Initializing Datasets...")
    # Load cached data if available, otherwise process from scratch
    train_dataset = ManufacturingDataset(split="train", load_cached_data=True)
    val_dataset = ManufacturingDataset(split="val", load_cached_data=True)

    # Create DataLoaders
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

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("Initializing Model...")
    model = HybridResFunnel().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    criterion = nn.BCEWithLogitsLoss()

    # --------------------------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------------------------
    trainer = Trainer(model, optimizer, scheduler, criterion, device, Config)

    # Train with early stopping
    trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS, patience=5)

    # --------------------------------------------------------------------------
    # 5. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Validation & Failure Analysis ---")

    # Load the best model checkpoint
    if os.path.exists(Config.MODEL_PATH):
        checkpoint = load_checkpoint(Config.MODEL_PATH, model, device=device)
        print(
            f"Loaded best model from epoch {checkpoint['epoch']} with AUC {checkpoint['best_auc']:.6f}"
        )
    else:
        print("Warning: No checkpoint found. Using current model state.")

    model.eval()

    # Collect predictions, targets, and inputs for analysis
    all_targets = []
    all_preds = []
    all_inputs_cont = []

    with torch.no_grad():
        for x_cont, x_cat, y in val_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            outputs = model(x_cont, x_cat)

            all_targets.append(y.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
            all_inputs_cont.append(x_cont.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_inputs_cont = np.concatenate(all_inputs_cont)

    # Calculate Final Metric
    val_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {val_auc}")

    # Failure Analysis: Correlation of Error Magnitude with Features
    errors = np.abs(all_targets - all_preds).flatten()

    # Feature names for continuous variables (f_00 to f_30, excluding f_27)
    feat_cols = [f"f_{i:02d}" for i in range(31) if i != 27]

    # Create DataFrame for correlation calculation
    # Note: all_inputs_cont is scaled, but correlation is invariant to linear scaling
    df_analysis = pd.DataFrame(all_inputs_cont, columns=feat_cols)
    df_analysis["error_magnitude"] = errors

    # Compute correlations
    correlations = (
        df_analysis.corr()["error_magnitude"]
        .drop("error_magnitude")
        .abs()
        .sort_values(ascending=False)
    )

    print("\nTop 5 Feature Correlations with Error Magnitude:")
    print(correlations.head(5))

    # --------------------------------------------------------------------------
    # 6. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9967793385748163

    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset = ManufacturingDataset(split="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Generate Predictions
        preds = predict(model, test_loader, device)

        # Create Submission File
        # We read sample_submission to ensure correct ID alignment
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

        # Verify length matches
        if len(preds) != len(sample_sub):
            print(
                f"Warning: Prediction length {len(preds)} does not match sample submission {len(sample_sub)}."
            )

        sample_sub["target"] = preds

        # Save
        sample_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation AUC ({val_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
