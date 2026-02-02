import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data import get_dataloaders
from library.model import AutoencodingHybridNet
from library.train import train_one_epoch, validate, predict_and_submit


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration
    # --------------------------------------------------------------------------
    # Using optimized settings from Config (Cite solution_lesson_node_00067)
    # Config.EPOCHS = 35
    # Config.SCHEDULER_STEP_SIZE = 10

    # Ensure we use the full dataset to maximize performance
    Config.DEBUG = False

    # --------------------------------------------------------------------------
    # 2. Setup & Data Loading
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load data (using cache if available)
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    model = AutoencodingHybridNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Aggressive decay to converge fast within fewer epochs
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    best_auc = 0.0

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)

    # --------------------------------------------------------------------------
    # 5. Final Evaluation & Failure Analysis
    # --------------------------------------------------------------------------
    # Load the best model state
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))
    model.eval()

    # Collect predictions, targets, and features for analysis
    all_preds = []
    all_targets = []
    all_features = []

    with torch.no_grad():
        for batch in val_loader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["target"].to(device)

            # Forward pass (inference only)
            cls_logits = model(continuous, sequence)
            preds = torch.sigmoid(cls_logits)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            # Store continuous features for correlation analysis
            all_features.append(continuous.cpu().numpy())

    # Flatten/Concatenate
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    all_features = np.concatenate(all_features, axis=0)

    # Compute Final Metric
    final_auc = compute_auc(all_targets, all_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation between Error and Input Features
    print("\n--- Failure Analysis ---")
    errors = np.abs(all_targets - all_preds)

    correlations = []
    # Calculate correlation for each continuous feature
    for i in range(all_features.shape[1]):
        feat_values = all_features[:, i]
        # Avoid warning if variance is 0
        if np.std(feat_values) > 1e-9 and np.std(errors) > 1e-9:
            corr = np.corrcoef(feat_values, errors)[0, 1]
        else:
            corr = 0.0
        correlations.append((f"f_{i:02d}", corr))

    # Sort by magnitude of correlation (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.4f}")

    # --------------------------------------------------------------------------
    # 6. Conditional Submission
    # --------------------------------------------------------------------------
    SUBMISSION_THRESHOLD = 0.9970005855169476

    if final_auc > SUBMISSION_THRESHOLD:
        predict_and_submit(model, test_loader, test_ids, device)
    else:
        print(f"\nMetric {final_auc} <= {SUBMISSION_THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
