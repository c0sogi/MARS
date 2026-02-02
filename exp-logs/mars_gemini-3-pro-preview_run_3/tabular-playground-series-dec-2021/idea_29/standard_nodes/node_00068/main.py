import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.utils import (
    seed_everything,
    get_device,
    generate_submission,
    feature_engineering,
)
from library.data_loader import get_dataloaders
from library.model import DeepParallelVectorDCNResNet
from library.train import train_one_epoch, validate, EarlyStopping


def get_feature_names(data_dir="./metadata"):
    """
    Reconstructs feature names to match the order in get_dataloaders.
    Logic must match library.data_loader.get_dataloaders exactly.
    """
    # Load a small sample to get columns
    df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))

    # Drop non-feature columns
    df = df.drop(columns=["Id", "Cover_Type"], errors="ignore")

    # Apply feature engineering
    df = feature_engineering(df)

    # Identify columns (Logic from data_loader.py)
    bin_cols = [c for c in df.columns if "Soil_Type" in c or "Wilderness_Area" in c]
    cont_cols = [c for c in df.columns if c not in bin_cols]

    # Final order
    return cont_cols + bin_cols


def perform_failure_analysis(model, val_loader, device, feature_names):
    """
    Calculates correlation between input features and prediction errors.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_inputs = []
    all_preds = []
    all_targets = []

    # Collect data
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_inputs.append(inputs.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate
    X_val = np.concatenate(all_inputs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Calculate Error (1 if wrong, 0 if right)
    errors = (y_pred != y_true).astype(int)

    # Calculate correlations
    correlations = []
    for i in range(X_val.shape[1]):
        # Handle constant features to avoid division by zero in correlation
        if np.std(X_val[:, i]) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(X_val[:, i], errors)[0, 1]

        feat_name = (
            feature_names[i]
            if feature_names and i < len(feature_names)
            else f"Feature_{i}"
        )
        correlations.append((feat_name, corr))

    # Sort by correlation (descending)
    correlations.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 Features correlated with Error:")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.4f}")


def main():
    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using defaults: batch_size=4096, cache_dir='./working/idea_29'
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(
        load_cached_data=True, batch_size=4096
    )

    # Get feature names for analysis
    feature_names = get_feature_names()

    # 3. Model Initialization
    model = DeepParallelVectorDCNResNet(
        input_dim=input_dim,
        num_classes=7,
        hidden_dim=512,
        num_cross_layers=3,
        num_res_blocks=4,
        dropout_rate=0.2,
    ).to(device)

    # 4. Training Configuration
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    early_stopping = EarlyStopping(patience=10)

    epochs = 60
    print(f"Starting training for {epochs} epochs...")

    # 5. Training Loop
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.6f}"
        )

        scheduler.step(val_acc)
        early_stopping(val_acc, model)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 6. Load Best Weights
    if early_stopping.best_model_state is not None:
        print("Loading best model weights...")
        model.load_state_dict(early_stopping.best_model_state)

    # 7. Final Validation Metric
    # Recalculate to ensure exact metric is printed
    final_val_loss, final_val_acc = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_val_acc}")

    # 8. Failure Analysis
    perform_failure_analysis(model, val_loader, device, feature_names)

    # 9. Submission
    THRESHOLD = 0.9625222222222222

    if final_val_acc > THRESHOLD:
        print(
            f"Validation accuracy {final_val_acc} > {THRESHOLD}. Generating submission..."
        )
        generate_submission(
            model,
            test_loader,
            test_ids,
            device,
            output_path="./submission/submission.csv",
        )
    else:
        print(
            f"Validation accuracy {final_val_acc} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
