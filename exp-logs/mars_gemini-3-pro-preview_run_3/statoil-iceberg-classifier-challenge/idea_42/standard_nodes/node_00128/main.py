import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import log_loss

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.utils import get_device, load_dataset
from library.data import IcebergDataset
from library.model import AAHACNN
from library.train import train_one_epoch, validate


def predict_with_model(model, loader, device):
    """
    Generate predictions using a specific model.
    Efficiently handles inference by disabling gradients and using eval mode.
    Works for both validation (img, ang, lbl) and test (img, ang, id) loaders.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # Access only the first two elements: images and angles
            # This makes it compatible with both train/val and test datasets
            images = batch[0].to(device)
            angles = batch[1].to(device)

            outputs = model(images, angles)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


def main():
    # 1. Initialization
    Config.setup()
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    # We load train and val separately to strictly adhere to the hold-out validation requirement
    print("Loading datasets...")
    X_train, ang_train, y_train = load_dataset("train", load_cached_data=True)
    X_val, ang_val, y_val = load_dataset("val", load_cached_data=True)
    X_test, ang_test, ids_test = load_dataset("test", load_cached_data=True)

    # Define Augmentations for Training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Create Datasets
    train_ds = IcebergDataset(
        X_train, ang_train, labels=y_train, transform=train_transform
    )
    val_ds = IcebergDataset(X_val, ang_val, labels=y_val, transform=None)
    test_ds = IcebergDataset(X_test, ang_test, ids=ids_test, transform=None)

    # Create DataLoaders
    # Pin memory enabled for faster host-to-device transfer
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Training Loop (Ensemble)
    # Train multiple models with different seeds to improve stability and performance
    n_seeds = 5
    model_paths = []

    print(f"Starting training with {n_seeds} seeds...")

    for i in range(n_seeds):
        seed = Config.SEED + i
        set_seed(seed)
        print(f"\n--- Training Model {i+1}/{n_seeds} (Seed {seed}) ---")

        model = AAHACNN().to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        save_path = os.path.join(Config.CHECKPOINT_DIR, f"model_seed_{seed}.pth")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, _ = validate(model, val_loader, criterion, device)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Finished. Best Val Loss: {best_val_loss:.6f}")
        model_paths.append(save_path)

    # 4. Validation Assessment
    print("\nPerforming Ensemble Validation...")
    val_preds_accum = np.zeros((len(y_val), 1))

    # Aggregate predictions from all models
    for path in model_paths:
        model = AAHACNN().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        preds = predict_with_model(model, val_loader, device)
        val_preds_accum += preds

    avg_val_preds = val_preds_accum / n_seeds
    avg_val_preds_flat = avg_val_preds.flatten()

    # Compute Final Metric
    final_metric = log_loss(y_val, avg_val_preds_flat, labels=[0, 1])
    # Print full precision as required
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(y_val - avg_val_preds_flat)

    # Calculate feature statistics for correlation
    # X_val shape is (N, 3, 75, 75). Channel 0 is HH, Channel 1 is HV.
    b1_mean = np.mean(X_val[:, 0, :, :], axis=(1, 2))
    b1_std = np.std(X_val[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X_val[:, 1, :, :], axis=(1, 2))
    b2_std = np.std(X_val[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": ang_val,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
        }
    )

    # Compute correlations
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission Generation
    threshold = 0.1806015565870406

    if final_metric < threshold:
        print(
            f"\nMetric {final_metric} meets criteria (< {threshold}). Generating submission..."
        )

        test_preds_accum = np.zeros((len(ids_test), 1))

        for path in model_paths:
            model = AAHACNN().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            preds = predict_with_model(model, test_loader, device)
            test_preds_accum += preds

        avg_test_preds = test_preds_accum / n_seeds
        avg_test_preds_flat = avg_test_preds.flatten()

        submission_df = pd.DataFrame(
            {"id": ids_test, "is_iceberg": avg_test_preds_flat}
        )

        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
        print(submission_df.head())
    else:
        print(
            f"\nMetric {final_metric} did not meet criteria (< {threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
