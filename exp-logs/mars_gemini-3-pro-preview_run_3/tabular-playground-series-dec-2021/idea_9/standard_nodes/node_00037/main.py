import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# Import from library
from library.config import (
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    NUM_WORKERS,
    HIDDEN_DIM,
    NUM_CROSS_LAYERS,
    DROPOUT_RATE,
    NUM_CLASSES,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
)
from library.utils import seed_everything, ModelCheckpoint
from library.model import ParallelDCNResNet
from library.data_loader import get_dataloaders
from library.train import train_one_epoch, validate, predict_and_submit


def main():
    # 1. Setup
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extended training to ensure convergence for standard ResNet
    FAST_EPOCHS = 50

    # 2. Data Loading
    # Using cached data if available for speed
    train_loader, val_loader, test_loader, input_dim, test_ids = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
    )

    # 3. Model Initialization
    model = ParallelDCNResNet(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_cross_layers=NUM_CROSS_LAYERS,
        dropout=DROPOUT_RATE,
    ).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    checkpoint = ModelCheckpoint(mode="max")
    patience_counter = 0
    best_val_acc = 0.0

    print(f"Starting training for {FAST_EPOCHS} epochs on {device}...")

    for epoch in range(FAST_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{FAST_EPOCHS} | Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        scheduler.step(val_acc)

        improved = checkpoint.step(val_acc, model)
        if improved:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint.save_best(MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 6. Final Evaluation & Failure Analysis
    print("Loading best model for evaluation...")
    model = checkpoint.load_best(model)
    model.eval()

    # Re-run validation to get exact metric and data for failure analysis
    all_preds = []
    all_targets = []
    all_inputs = []

    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            running_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            # Store for failure analysis
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    final_metric = running_correct / total_samples
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    X_val = np.concatenate(all_inputs, axis=0)
    y_val = np.concatenate(all_targets, axis=0)
    preds_val = np.concatenate(all_preds, axis=0)

    errors = (preds_val != y_val).astype(int)  # 1 if error, 0 if correct

    # Calculate correlation between each feature and the error
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feat_col = X_val[:, i]
        # Check for constant columns to avoid warnings
        if np.std(feat_col) < 1e-9 or np.std(errors) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, errors)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features correlated with Error:")
    for idx, corr in correlations[:10]:
        print(f"Feature Index {idx}: Correlation {corr:.4f}")

    # 7. Conditional Submission
    THRESHOLD = 0.9625041666666667
    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric {final_metric} > {THRESHOLD}. Generating submission..."
        )
        predict_and_submit(model, test_loader, test_ids, device, SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric {final_metric} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
