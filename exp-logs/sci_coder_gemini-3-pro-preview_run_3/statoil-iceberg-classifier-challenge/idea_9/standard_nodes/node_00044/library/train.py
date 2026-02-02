import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.data_loader import load_and_process_data, IcebergDataset
from library.model import SimpleCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, angles, labels in loader:
        inputs = inputs.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        outputs = model(inputs, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, angles, labels in loader:
            inputs = inputs.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def predict(model, loader, device):
    """
    Generates predictions for a given loader.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            angles = batch[1].to(device)

            outputs = model(inputs, angles)
            probs = torch.sigmoid(outputs)
            preds.append(probs.cpu().numpy())

    return np.concatenate(preds, axis=0)


def run_fold(fold, train_loader, val_loader, device):
    """
    Executes the training pipeline for a single fold.
    """
    print(f"\nStarting Fold {fold}/{Config.N_FOLDS - 1}")

    # Initialize model
    model = SimpleCNN().to(device)

    # Optimizer: Adam with constant LR and Weight Decay
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss: BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = Config.get_checkpoint_path(fold)

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Fold {fold} | Epoch {epoch + 1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model.state_dict(), checkpoint_path)
            # print(f"  New best model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch + 1}")
                break

    return best_val_loss


def train_and_predict():
    """
    Main function to run 5-Fold CV training and generate submission.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    # We load cached data. If not present, it will be processed.
    data = load_and_process_data(load_cached_data=True)

    # Combine Train and Val for Cross-Validation
    # The provided data loader splits based on metadata, but for 5-fold CV
    # we need the full training set to re-split.
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)
    angles_full = np.concatenate([data["angles_train"], data["angles_val"]], axis=0)

    # Test Data
    X_test = data["X_test"]
    angles_test = data["angles_test"]
    ids_test = data["ids_test"]

    # 2. Prepare Test Loader (No shuffle, no transform)
    test_dataset = IcebergDataset(X_test, angles_test, y=None, transform=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # 3. Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_scores = []
    test_preds_sum = np.zeros((len(X_test), 1))

    # Augmentations for training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        # Split data
        X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]
        angles_train_fold, angles_val_fold = (
            angles_full[train_idx],
            angles_full[val_idx],
        )

        # Create Datasets
        train_ds = IcebergDataset(
            X_train_fold, angles_train_fold, y_train_fold, transform=train_transform
        )
        val_ds = IcebergDataset(X_val_fold, angles_val_fold, y_val_fold, transform=None)

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        # Train Fold
        best_loss = run_fold(fold, train_loader, val_loader, device)
        fold_scores.append(best_loss)

        # Load Best Model for Inference
        model = SimpleCNN().to(device)
        checkpoint_path = Config.get_checkpoint_path(fold)
        load_checkpoint(checkpoint_path, model, device=Config.DEVICE)

        # Predict on Test Set
        fold_preds = predict(model, test_loader, device)
        test_preds_sum += fold_preds

        # Clean up
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()

    avg_cv_loss = np.mean(fold_scores)
    print(f"\n5-Fold CV Complete. Average Validation Loss: {avg_cv_loss:.8f}")

    # 4. Generate Submission
    # Average predictions across folds
    avg_preds = test_preds_sum / Config.N_FOLDS

    # Create DataFrame
    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds.flatten()})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Done.")
