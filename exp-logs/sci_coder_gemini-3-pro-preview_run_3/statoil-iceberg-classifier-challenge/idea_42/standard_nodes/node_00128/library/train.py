import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torchvision import transforms

from library.config import Config, set_seed
from library.utils import get_device, load_dataset
from library.data import IcebergDataset
from library.model import AAHACNN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape (B, 1)

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and log loss.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate Log Loss (clipping handled by sklearn usually, but good to be safe)
    # Log loss requires probabilities
    metric_log_loss = log_loss(all_labels, all_preds, labels=[0, 1])

    return epoch_loss, metric_log_loss


def predict_test(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, angles, ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


def run_kfold(n_folds=Config.N_FOLDS, epochs=Config.NUM_EPOCHS, debug=Config.DEBUG):
    """
    Executes K-Fold Cross Validation training and generates submission.
    """
    set_seed(Config.SEED)
    device = get_device()

    print(
        f"Starting execution with Device: {device}, Folds: {n_folds}, Epochs: {epochs}, Debug: {debug}"
    )

    # 1. Load and Merge Data
    # We merge train and val splits from metadata to perform our own K-Fold
    X_train_part, ang_train_part, y_train_part = load_dataset("train")
    X_val_part, ang_val_part, y_val_part = load_dataset("val")

    X_full = np.concatenate([X_train_part, X_val_part], axis=0)
    ang_full = np.concatenate([ang_train_part, ang_val_part], axis=0)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)

    # Load Test Data
    X_test, ang_test, ids_test = load_dataset("test")

    # Handle Debug Mode
    if debug:
        print("Debug mode enabled: Slicing datasets.")
        subset_size = 64
        X_full = X_full[:subset_size]
        ang_full = ang_full[:subset_size]
        y_full = y_full[:subset_size]
        X_test = X_test[:subset_size]
        ang_test = ang_test[:subset_size]
        ids_test = ids_test[:subset_size]

    # Prepare Test Loader (Fixed)
    test_dataset = IcebergDataset(X_test, ang_test, ids=ids_test, transform=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # Array to store test predictions from each fold
    test_predictions_sum = np.zeros((len(X_test), 1))

    # 2. K-Fold Loop
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    # Define Transforms
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n{'='*20} Fold {fold+1}/{n_folds} {'='*20}")

        # Split Data
        X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
        ang_train_fold, ang_val_fold = ang_full[train_idx], ang_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(
            X_train_fold, ang_train_fold, labels=y_train_fold, transform=train_transform
        )
        val_ds = IcebergDataset(
            X_val_fold, ang_val_fold, labels=y_val_fold, transform=None
        )

        # Create Loaders
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

        # Initialize Model
        model = AAHACNN().to(device)

        # Optimizer & Criterion
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop with Early Stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_log_loss = validate(model, val_loader, criterion, device)

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val LogLoss: {val_log_loss:.10f} | "
                f"Time: {elapsed:.2f}s"
            )

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                # print(f"  -> Model saved (Best Val Loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load Best Model for Inference
        print(f"Loading best model for Fold {fold+1}...")
        model.load_state_dict(torch.load(best_model_path))

        # Predict on Test Set
        fold_preds = predict_test(model, test_loader, device)
        test_predictions_sum += fold_preds

    # 3. Generate Submission
    print("\nGenerating submission...")
    avg_preds = test_predictions_sum / n_folds

    # Flatten predictions
    avg_preds = avg_preds.flatten()

    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(submission_df.head())
