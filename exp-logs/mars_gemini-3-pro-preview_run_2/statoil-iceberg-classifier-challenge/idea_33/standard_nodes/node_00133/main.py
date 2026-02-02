import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
import copy

from library.config import Config, IcebergDataset
from library.model import MDSWBN
from library.data_loader import get_data, get_test_loader
from library.utils import seed_everything
from library.train import train_one_epoch, validate


def main():
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Configuration for Fast Baseline
    EPOCHS = 20
    BATCH_SIZE = Config.BATCH_SIZE
    DEVICE = Config.DEVICE

    print(f"Running on device: {DEVICE}")

    # 1. Load Metadata to define splits
    print("Loading metadata...")
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")

    if not os.path.exists(train_meta_path) or not os.path.exists(val_meta_path):
        raise FileNotFoundError("Metadata files not found.")

    df_train_meta = pd.read_csv(train_meta_path)
    df_val_meta = pd.read_csv(val_meta_path)

    n_train = len(df_train_meta)
    n_val = len(df_val_meta)
    print(f"Metadata counts - Train: {n_train}, Val: {n_val}")

    # 2. Load Data
    # get_data returns the combined training data (Train + Val) as processed by load_and_process_data
    print("Loading and processing data...")
    X_all, y_all, inc_all, X_test, inc_test, test_ids = get_data(debug=False)

    # Verify shapes
    if len(X_all) != n_train + n_val:
        print(
            f"Warning: Data length {len(X_all)} does not match metadata sum {n_train + n_val}"
        )

    # 3. Split Data
    # load_and_process_data concatenates train then val, so we slice accordingly
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    inc_train = inc_all[:n_train]

    X_val = X_all[n_train:]
    y_val = y_all[n_train:]
    inc_val = inc_all[n_train:]

    # 4. Create Datasets and Loaders
    train_ds = IcebergDataset(X_train, inc_train, y_train, transform=True)
    val_ds = IcebergDataset(X_val, inc_val, y_val, transform=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Model Initialization
    model = MDSWBN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 6. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    print("Training complete.")

    # 7. Final Evaluation
    # Load best weights
    model.load_state_dict(best_model_wts)
    model.eval()

    val_preds = []
    val_targets = []

    with torch.no_grad():
        for images, angles, labels in val_loader:
            images = images.to(DEVICE)
            angles = angles.to(DEVICE)
            outputs = model(images, angles)

            val_preds.extend(outputs.cpu().numpy().flatten())
            val_targets.extend(labels.cpu().numpy().flatten())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate Metric
    final_metric = log_loss(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 8. Failure Analysis
    print("\nFailure Analysis:")
    errors = np.abs(val_targets - val_preds)

    # Calculate image statistics for correlation
    # X_val shape: (N, 3, 75, 75) -> (N, C, H, W)
    # Channel 0: Band 1, Channel 1: Band 2
    b1_means = np.mean(X_val[:, 0, :, :], axis=(1, 2))
    b2_means = np.mean(X_val[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": inc_val,
            "b1_mean": b1_means,
            "b2_mean": b2_means,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error and Features:")
    print(correlations)

    # 9. Submission
    THRESHOLD = 0.15744295919935183

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        test_loader = get_test_loader(X_test, inc_test, batch_size=BATCH_SIZE)
        test_preds = []

        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(DEVICE)
                angles = angles.to(DEVICE)
                outputs = model(images, angles)
                test_preds.extend(outputs.cpu().numpy().flatten())

        test_preds = np.array(test_preds)

        # Save submission
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        sub_df = pd.DataFrame({"id": test_ids, "is_iceberg": test_preds})
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
