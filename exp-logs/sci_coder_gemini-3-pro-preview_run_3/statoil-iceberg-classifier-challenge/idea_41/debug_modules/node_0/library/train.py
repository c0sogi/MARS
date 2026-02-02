import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import set_seed, setup_logger
from library.data import _get_data_splits, IcebergDataset
from library.model import PDPH_SE_CNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for inputs, angles, labels in loader:
        inputs = inputs.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

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
    Returns average loss and true/pred lists for metric calculation.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, angles, labels in loader:
            inputs = inputs.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(inputs, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid for probability output
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(all_preds), np.array(all_targets)


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, angles, _ in loader:
            inputs = inputs.to(device)
            angles = angles.to(device)

            outputs = model(inputs, angles)
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())

    return np.array(all_preds)


def run_training():
    """
    Main execution function for 5-Fold Cross-Validation training.
    """
    logger = setup_logger(os.path.join(Config.WORKING_DIR, "train.log"))
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    logger.info("Starting PDPH-SE-CNN Training Pipeline")
    logger.info(f"Device: {device}")

    # 1. Load Data
    # We use the internal helper to get raw arrays, then merge for K-Fold
    data_map = _get_data_splits(load_cached_data=True)

    X_train_part = data_map["X_train"]
    y_train_part = data_map["y_train"]
    ang_train_part = data_map["angles_train"]
    ids_train_part = data_map["ids_train"]

    X_val_part = data_map["X_val"]
    y_val_part = data_map["y_val"]
    ang_val_part = data_map["angles_val"]
    ids_val_part = data_map["ids_val"]

    # Merge for Cross-Validation
    X_full = np.concatenate([X_train_part, X_val_part], axis=0)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)
    ang_full = np.concatenate([ang_train_part, ang_val_part], axis=0)
    ids_full = np.concatenate([ids_train_part, ids_val_part], axis=0)

    X_test = data_map["X_test"]
    ang_test = data_map["angles_test"]
    ids_test = data_map["ids_test"]

    # Debug Subset
    if Config.DEBUG:
        logger.info(f"DEBUG Mode: Subsetting to {Config.DEBUG_SAMPLE_SIZE} samples")
        limit = min(Config.DEBUG_SAMPLE_SIZE, len(X_full))
        X_full = X_full[:limit]
        y_full = y_full[:limit]
        ang_full = ang_full[:limit]
        ids_full = ids_full[:limit]

        X_test = X_test[:limit]
        ang_test = ang_test[:limit]
        ids_test = ids_test[:limit]

    logger.info(f"Total Training Samples: {len(X_full)}")
    logger.info(f"Total Test Samples: {len(X_test)}")

    # 2. Prepare Test Loader (Common for all folds)
    test_dataset = IcebergDataset(
        X_test, ang_test, y=None, transform=None, ids=ids_test
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # 3. K-Fold Cross Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_test_preds = []
    checkpoints_dir = os.path.join(Config.WORKING_DIR, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Define Transforms
    train_transform = None
    if Config.USE_AUGMENTATION:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=Config.HORIZONTAL_FLIP_PROB),
                transforms.RandomVerticalFlip(p=Config.VERTICAL_FLIP_PROB),
            ]
        )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        logger.info(f"\n{'='*20} Fold {fold+1}/{Config.N_FOLDS} {'='*20}")

        # Split Data
        X_tr, X_va = X_full[train_idx], X_full[val_idx]
        y_tr, y_va = y_full[train_idx], y_full[val_idx]
        ang_tr, ang_va = ang_full[train_idx], ang_full[val_idx]
        ids_tr, ids_va = ids_full[train_idx], ids_full[val_idx]

        # Create Datasets/Loaders
        train_ds = IcebergDataset(
            X_tr, ang_tr, y_tr, transform=train_transform, ids=ids_tr
        )
        val_ds = IcebergDataset(X_va, ang_va, y_va, transform=None, ids=ids_va)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        # Initialize Model
        model = PDPH_SE_CNN().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Training Loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(checkpoints_dir, f"model_fold_{fold}.pth")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_preds, val_targets = validate(
                model, val_loader, criterion, device
            )

            # Metric: Log Loss
            # Clip predictions for stability in log_loss calculation
            val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
            val_log_loss = log_loss(val_targets, val_preds_clipped)

            logger.info(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val LogLoss: {val_log_loss:.10f}"
            )

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                # logger.info("  New best model saved.")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    logger.info(f"  Early stopping triggered at epoch {epoch+1}")
                    break

        # Load Best Model for Inference
        logger.info(
            f"Loading best model for Fold {fold+1} with Val Loss: {best_val_loss:.6f}"
        )
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Predict on Test Set
        fold_preds = predict(model, test_loader, device)
        fold_test_preds.append(fold_preds)

    # 4. Ensemble and Submission
    logger.info("\nGenerating Final Submission...")

    # Average predictions across folds
    avg_preds = np.mean(fold_test_preds, axis=0)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Validate format against sample
    sample_df = pd.read_csv(Config.SAMPLE_SUBMISSION)
    logger.info(
        f"Submission shape: {submission_df.shape}, Sample shape: {sample_df.shape}"
    )
    logger.info("Head of submission:")
    logger.info(submission_df.head().to_string())
