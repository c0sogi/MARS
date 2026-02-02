import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import process_and_cache_data, IcebergDataset
from library.model import ShadowAwareWideBodyNet, train_one_epoch, validate, predict


def train_fold(fold_idx, train_loader, val_loader, device):
    """
    Manages the training lifecycle for a single fold, including optimization,
    scheduling, and early stopping.

    Args:
        fold_idx (int): Index of the current fold.
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.
        device (str): Device to train on ('cuda' or 'cpu').

    Returns:
        dict: The state dictionary of the best model found during training.
        float: The best validation loss achieved.
    """
    print(f"\n=== Fold {fold_idx + 1}/{Config.NUM_FOLDS} ===")

    # Initialize Model
    model = ShadowAwareWideBodyNet().to(device)

    # Setup Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Early Stopping Tracking
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        # Forward and Backward Pass
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        val_loss, val_probs, val_targets = validate(
            model, val_loader, criterion, device
        )

        # Learning Rate Scheduling
        scheduler.step(val_loss)

        # Check for Improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Logging (Full Precision)
        # Log every 5 epochs or if patience was reset (improvement found)
        if epoch % 5 == 0 or patience_counter == 0:
            print(f"Epoch {epoch+1}: Train Loss: {train_loss}, Val Loss: {val_loss}")

        # Early Stopping Trigger
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return best_model_wts, best_val_loss


def run_training_pipeline():
    """
    Executes the full Stratified K-Fold Cross-Validation pipeline.
    Loads data, trains models for each fold, generates ensemble predictions,
    and saves the final submission.
    """
    # 1. Setup
    set_seed(Config.SEED)
    logger = setup_logger(os.path.join(Config.WORK_DIR, "train.log"))

    # 2. Data Loading & Caching
    # This ensures data is processed and cached if not already present
    data = process_and_cache_data(load_cached_data=True)

    # 3. Prepare Data for Cross-Validation
    # Concatenate Train and Validation sets from metadata to use full labeled data for CV
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    ang_full = np.concatenate([data["ang_train"], data["ang_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)
    ids_full = np.concatenate([data["ids_train"], data["ids_val"]], axis=0)

    X_test = data["X_test"]
    ang_test = data["ang_test"]
    ids_test = data["ids_test"]

    # 4. Prepare Test Loader (for Inference)
    test_dataset = IcebergDataset(
        X_test, ang_test, labels=None, ids=ids_test, transform=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Stratified K-Fold Loop
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_scores = []
    test_preds_accum = np.zeros(len(ids_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        # Create Fold Datasets
        train_ds = IcebergDataset(
            X_full[train_idx],
            ang_full[train_idx],
            y_full[train_idx],
            ids_full[train_idx],
            transform=True,
        )
        val_ds = IcebergDataset(
            X_full[val_idx],
            ang_full[val_idx],
            y_full[val_idx],
            ids_full[val_idx],
            transform=False,
        )

        # Create Fold Loaders
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

        # Train the Fold
        best_wts, best_loss = train_fold(fold, train_loader, val_loader, Config.DEVICE)
        fold_scores.append(best_loss)

        # Save Best Model for this Fold
        model_path = os.path.join(Config.MODEL_DIR, f"sa_wbn_fold_{fold}.pth")
        torch.save(best_wts, model_path)
        print(f"Saved model to {model_path}")

        # Inference on Test Set (Ensemble Component)
        # Re-initialize model to load best weights cleanly
        model = ShadowAwareWideBodyNet().to(Config.DEVICE)
        model.load_state_dict(best_wts)
        model.eval()

        _, fold_test_preds = predict(model, test_loader, Config.DEVICE)
        test_preds_accum += fold_test_preds

        # Clean up to save memory
        del model, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()

    # 6. Results & Submission
    # Average predictions across folds
    avg_test_preds = test_preds_accum / Config.NUM_FOLDS

    print("\n=== Cross-Validation Results ===")
    for i, score in enumerate(fold_scores):
        print(f"Fold {i+1}: {score}")
    print(f"Average Log Loss: {np.mean(fold_scores)}")

    # Save Submission File
    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds})
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
