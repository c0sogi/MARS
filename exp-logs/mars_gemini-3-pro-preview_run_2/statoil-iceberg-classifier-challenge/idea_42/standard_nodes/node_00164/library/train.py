import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import (
    DEVICE,
    BATCH_SIZE,
    MAX_EPOCHS,
    LEARNING_RATE,
    MODEL_CHECKPOINT_TEMPLATE,
    CACHE_PATH,
    SUBMISSION_PATH,
    NUM_WORKERS,
    SEED,
    DEBUG_DATA_LIMIT,
    N_FOLDS,
    PATIENCE,
)
from library.utils import (
    seed_everything,
    calculate_log_loss,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
)
from library.data_loader import load_data, IcebergDataset
from library.model import DN_WBN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, angles, labels) in enumerate(loader):
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

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
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    # Calculate metric using the utility function (Log Loss)
    metric_score = calculate_log_loss(all_labels, all_preds)

    return epoch_loss, metric_score


def run_fold(fold_idx, train_loader, val_loader, device):
    """
    Runs the training and validation for a single fold.
    """
    print(f"\nStarting Fold {fold_idx}...")

    # Initialize Model
    model = DN_WBN().to(device)

    # Optimizer: Adam (Low and Slow strategy)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Loss Function: BCELoss (model returns sigmoid probabilities)
    criterion = nn.BCELoss()

    # Early Stopping
    checkpoint_path = MODEL_CHECKPOINT_TEMPLATE.format(fold_idx)
    early_stopping = EarlyStopping(
        patience=PATIENCE, verbose=True, path=checkpoint_path
    )

    for epoch in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        # Update scheduler based on validation loss
        scheduler.step(val_loss)

        # Print metrics with full precision
        print(
            f"Fold {fold_idx} | Epoch {epoch+1}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.15f} | "
            f"Val Metric (LogLoss): {val_metric:.15f}"
        )

        # Check Early Stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model weights
    model.load_state_dict(early_stopping.best_model_state)
    return model


def predict_test(model, test_loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []
    ids = []

    with torch.no_grad():
        for batch in test_loader:
            # Test loader returns (images, angles, ids)
            images, angles, batch_ids = batch
            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            preds.extend(outputs.cpu().numpy())
            ids.extend(batch_ids)

    return np.array(ids), np.array(preds)


def run_training_pipeline():
    """
    Main function to execute the Stratified 5-Fold Cross-Validation pipeline.
    """
    seed_everything(SEED)

    # 1. Load Data (Triggers caching)
    # We ignore the returned train/val loaders because we merge them for 5-Fold CV.
    # We keep the test_loader (which is consistent).
    _, _, test_loader_initial = load_data(load_cached_data=True)

    # 2. Load Raw Arrays from Cache to perform Stratified CV
    print(f"Loading cached data from {CACHE_PATH} for Cross-Validation...")
    cached = np.load(CACHE_PATH, allow_pickle=True)

    # Extract Train and Val sets
    train_images = cached["train_images"]
    train_angles = cached["train_angles"]
    train_labels = cached["train_labels"]
    train_ids = cached["train_ids"]

    val_images = cached["val_images"]
    val_angles = cached["val_angles"]
    val_labels = cached["val_labels"]
    val_ids = cached["val_ids"]

    # Extract Global Stats and Angle Mean
    global_min = cached["global_min"]
    global_max = cached["global_max"]
    angle_mean = float(cached["angle_mean"])
    global_stats = (global_min, global_max)

    # Combine Train and Val for Stratified CV
    X_img = np.concatenate([train_images, val_images], axis=0)
    X_ang = np.concatenate([train_angles, val_angles], axis=0)
    y = np.concatenate([train_labels, val_labels], axis=0)
    ids = np.concatenate([train_ids, val_ids], axis=0)

    # Apply Debug Limit if set
    if DEBUG_DATA_LIMIT is not None:
        print(f"Applying DEBUG_DATA_LIMIT: {DEBUG_DATA_LIMIT}")
        limit = min(DEBUG_DATA_LIMIT, len(y))
        X_img = X_img[:limit]
        X_ang = X_ang[:limit]
        y = y[:limit]
        ids = ids[:limit]

    print(f"Total Combined Training Data: {len(y)} samples")

    # 3. Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Prepare Test Loader
    test_loader = test_loader_initial

    # Store predictions from each fold
    fold_preds = []
    test_ids = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_img, y)):
        # Create Datasets
        train_ds = IcebergDataset(
            X_img[train_idx],
            X_ang[train_idx],
            y[train_idx],
            ids[train_idx],
            global_stats=global_stats,
            transform=True,
            angle_mean=angle_mean,
        )
        val_ds = IcebergDataset(
            X_img[val_idx],
            X_ang[val_idx],
            y[val_idx],
            ids[val_idx],
            global_stats=global_stats,
            transform=False,
            angle_mean=angle_mean,
        )

        # Create Loaders
        # Use generator for reproducibility in shuffling
        g = torch.Generator()
        g.manual_seed(SEED)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_everything,
            generator=g,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_everything,
            pin_memory=True,
        )

        # Run Fold
        best_model = run_fold(fold, train_loader, val_loader, DEVICE)

        # Predict on Test Set
        print(f"Generating predictions for Fold {fold}...")
        current_ids, current_preds = predict_test(best_model, test_loader, DEVICE)

        if test_ids is None:
            test_ids = current_ids
        else:
            # Verify alignment
            if not np.array_equal(test_ids, current_ids):
                raise ValueError("Test IDs mismatch between folds!")

        fold_preds.append(current_preds)

        # Clear memory
        del best_model, train_ds, val_ds, train_loader, val_loader
        torch.cuda.empty_cache()

    # 4. Ensemble Predictions
    print("Ensembling predictions...")
    fold_preds = np.array(fold_preds)  # Shape: (N_folds, N_test)
    avg_preds = np.mean(fold_preds, axis=0)

    # 5. Create Submission
    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Done.")
