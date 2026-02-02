import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import (
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_FOLDS,
    NUM_EPOCHS,
    PATIENCE,
    WORKING_DIR,
    SUBMISSION_PATH,
    DEVICE,
    NUM_WORKERS,
)
from library.utils import EarlyStopping, set_seed
from library.model import RDPWBN, IcebergDataset, load_and_process_data


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for imgs, incs, labels in loader:
        imgs = imgs.to(device)
        incs = incs.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(imgs, incs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for imgs, incs, labels in loader:
            imgs = imgs.to(device)
            incs = incs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(imgs, incs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)

            # Binary classification accuracy
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == (labels > 0.5)).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / dataset_size
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def predict_test(model, X_test, inc_test, device=DEVICE):
    """
    Generates predictions for the test set.
    """
    model.eval()
    test_ds = IcebergDataset(X_test, None, inc_test, transform=False)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    preds = []
    with torch.no_grad():
        for imgs, incs in test_loader:
            imgs = imgs.to(device)
            incs = incs.to(device)
            outputs = model(imgs, incs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds.extend(probs)

    return np.array(preds)


def run_fold(fold, X_train, y_train, inc_train, X_val, y_val, inc_val, device=DEVICE):
    """
    Executes the training loop for a specific cross-validation fold.
    """
    print(f"\n=== Fold {fold} ===")

    # Create Datasets
    # Train set gets augmentation (transform=True)
    train_ds = IcebergDataset(X_train, y_train, inc_train, transform=True)
    val_ds = IcebergDataset(X_val, y_val, inc_val, transform=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Initialize Model
    model = RDPWBN().to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping
    checkpoint_path = os.path.join(WORKING_DIR, f"model_fold_{fold}.pth")
    early_stopping = EarlyStopping(
        patience=PATIENCE, verbose=True, path=checkpoint_path
    )

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss} - Val Acc: {val_acc}"
        )

        # Scheduler Step
        scheduler.step(val_loss)

        # Early Stopping Step
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model state saved by EarlyStopping
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def train_cross_validation(load_cached_data=True):
    """
    Main function to run Stratified 5-Fold Cross-Validation and generate submission.
    """
    set_seed(SEED)

    # Load and Process Data (uses library function with caching)
    X, y, inc, X_test, inc_test, test_ids = load_and_process_data(
        load_cached_data=load_cached_data
    )

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # Accumulator for test predictions
    test_preds_accum = np.zeros(len(test_ids))

    print(f"Starting Stratified {NUM_FOLDS}-Fold Cross-Validation on {DEVICE}...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split Data
        X_tr, y_tr, inc_tr = X[train_idx], y[train_idx], inc[train_idx]
        X_val, y_val, inc_val = X[val_idx], y[val_idx], inc[val_idx]

        # Run Training for this Fold
        model = run_fold(fold, X_tr, y_tr, inc_tr, X_val, y_val, inc_val, device=DEVICE)

        # Inference on Test Set
        print(f"Generating predictions for Fold {fold}...")
        fold_preds = predict_test(model, X_test, inc_test, device=DEVICE)
        test_preds_accum += fold_preds

    # Average Predictions
    final_preds = test_preds_accum / NUM_FOLDS

    # Save Submission
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": final_preds})
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
