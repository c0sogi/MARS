import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.data import prepare_data, IcebergDataset, get_transforms
from library.model import SimpleCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def run_fold(fold_idx, train_loader, val_loader):
    """
    Runs the training loop for a single fold.
    """
    print(f"\n--- Starting Fold {fold_idx} ---")
    device = Config.DEVICE

    # Initialize Model
    model = SimpleCNN().to(device)

    # Optimizer (Adam with constant LR) and Loss
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Fold {fold_idx} | Epoch {epoch + 1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Time: {elapsed:.2f}s"
        )

        # Checkpoint Logic
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_val_loss,
                "fold": fold_idx,
            },
            is_best,
            fold_idx,
        )

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1} for Fold {fold_idx}")
            break

    print(f"Fold {fold_idx} Best Validation Loss: {best_val_loss}")
    return best_val_loss


def predict_and_submit(test_data):
    """
    Generates predictions using the ensemble of trained models and saves submission.
    """
    print("\n--- Starting Inference and Submission Generation ---")
    device = Config.DEVICE
    X_test, angle_test, ids_test = test_data

    # Create Test Loader
    test_dataset = IcebergDataset(
        X_test, angle_test, y=None, ids=ids_test, transform=get_transforms("test")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Array to store sum of predictions (for averaging)
    # Using sigmoid to convert logits to probabilities
    avg_preds = np.zeros((len(ids_test), 1))

    for fold_idx in range(Config.N_FOLDS):
        print(f"Predicting with model from Fold {fold_idx}...")

        # Load Model
        model = SimpleCNN().to(device)
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )
        load_checkpoint(checkpoint_path, model)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, angles, _ in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                logits = model(images, angles)
                probs = torch.sigmoid(logits)
                fold_preds.append(probs.cpu().numpy())

        fold_preds = np.concatenate(fold_preds, axis=0)
        avg_preds += fold_preds

    # Average predictions
    avg_preds /= Config.N_FOLDS
    avg_preds = avg_preds.flatten()

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission generation complete.")


def run_training_process():
    """
    Main entry point to run the 5-Fold CV training and inference.
    """
    set_seed(Config.SEED)

    # 1. Load Data
    # We load cached data. If cache doesn't exist, prepare_data creates it.
    (train_data_split, val_data_split, test_data) = prepare_data(load_cached_data=True)

    # 2. Merge Train and Val splits to perform K-Fold CV
    X_train, angle_train, y_train, ids_train = train_data_split
    X_val, angle_val, y_val, ids_val = val_data_split

    X_full = np.concatenate([X_train, X_val], axis=0)
    angle_full = np.concatenate([angle_train, angle_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    ids_full = np.concatenate([ids_train, ids_val], axis=0)

    print(f"Total Training Samples for CV: {len(y_full)}")

    # 3. Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        # Split Data
        X_tr, X_va = X_full[train_idx], X_full[val_idx]
        angle_tr, angle_va = angle_full[train_idx], angle_full[val_idx]
        y_tr, y_va = y_full[train_idx], y_full[val_idx]
        ids_tr, ids_va = ids_full[train_idx], ids_full[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(
            X_tr, angle_tr, y_tr, ids_tr, transform=get_transforms("train")
        )
        val_ds = IcebergDataset(
            X_va, angle_va, y_va, ids_va, transform=get_transforms("val")
        )

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Run Fold
        best_loss = run_fold(fold_idx, train_loader, val_loader)
        fold_scores.append(best_loss)

    # 4. Report CV Results
    print("\n--- Cross-Validation Results ---")
    for i, score in enumerate(fold_scores):
        print(f"Fold {i}: {score}")
    print(f"Average Log Loss: {np.mean(fold_scores)}")

    # 5. Inference
    predict_and_submit(test_data)
