import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library import config, utils, data_loader, model

# Initialize logger
logger = utils.get_logger("train")


def train_one_epoch(model_inst, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model_inst.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for batch in loader:
        # Unpack batch
        # IcebergDataset returns (img, inc, label) when labels are present
        inputs, inc_angles, labels = batch

        inputs = inputs.to(device)
        inc_angles = inc_angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model_inst(inputs, inc_angles)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model_inst, loader, criterion, device):
    """
    Validates the model for one epoch.
    """
    model_inst.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for batch in loader:
            inputs, inc_angles, labels = batch

            inputs = inputs.to(device)
            inc_angles = inc_angles.to(device)
            labels = labels.to(device)

            outputs = model_inst(inputs, inc_angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def predict(model_inst, loader, device):
    """
    Generates predictions for a given loader.
    """
    model_inst.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            # Handle cases where loader returns (img, inc) or (img, inc, label)
            if len(batch) == 2:
                inputs, inc_angles = batch
            else:
                inputs, inc_angles, _ = batch

            inputs = inputs.to(device)
            inc_angles = inc_angles.to(device)

            outputs = model_inst(inputs, inc_angles)
            probs = torch.sigmoid(outputs)
            preds.append(probs.cpu().numpy())

    return np.concatenate(preds, axis=0)


def run_fold(fold_idx, train_loader, val_loader, device):
    """
    Executes the training and validation loop for a single fold.
    """
    logger.info(f"Starting Fold {fold_idx}")

    # Initialize Model
    net = model.WBDIN().to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        min_lr=config.MIN_LR,
    )

    # Early Stopping Tracking
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(net.state_dict())
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(net, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_loss)

        # Log Metrics (Full Precision)
        logger.info(
            f"Fold {fold_idx} Epoch {epoch+1}/{config.EPOCHS} - "
            f"Train Loss: {train_loss:.10f} - Val Loss: {val_loss:.10f}"
        )

        # Check Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(net.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Save Best Model
    save_path = os.path.join(config.WORKING_DIR, f"model_fold_{fold_idx}.pth")
    torch.save(best_model_wts, save_path)
    logger.info(f"Fold {fold_idx} finished. Best Val Loss: {best_loss:.10f}")

    return best_loss


def run_training():
    """
    Main pipeline function.
    Loads data, performs 5-Fold CV, trains models, and generates submission.
    """
    utils.seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Processed Data
    # We use the provided data_loader which handles caching internally
    data = data_loader.process_data(load_cached_data=True)

    # 2. Prepare Data for Cross-Validation
    # Combine the pre-split train and val sets to perform our own Stratified K-Fold
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)
    inc_full = np.concatenate([data["inc_train"], data["inc_val"]], axis=0)

    scaling_stats = (data["ch_mins"], data["ch_maxs"])

    # 3. Prepare Test Data
    X_test = data["X_test"]
    inc_test = data["inc_test"]

    # Apply Debug Trimming if enabled
    if config.DEBUG:
        logger.info(f"Debug mode: trimming datasets to {config.DEBUG_SIZE} samples")
        limit = min(config.DEBUG_SIZE, len(X_full))
        X_full = X_full[:limit]
        y_full = y_full[:limit]
        inc_full = inc_full[:limit]

        limit_test = min(config.DEBUG_SIZE, len(X_test))
        X_test = X_test[:limit_test]
        inc_test = inc_test[:limit_test]

    test_ds = data_loader.IcebergDataset(
        X_test,
        inc_test,
        labels=None,
        transform=data_loader.get_transforms(augment=False),
        scaling_stats=scaling_stats,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 4. Stratified K-Fold Loop
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Array to accumulate test predictions
    test_predictions_sum = np.zeros((len(X_test), 1))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        # Split Data
        X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]
        inc_train_fold, inc_val_fold = inc_full[train_idx], inc_full[val_idx]

        # Create Datasets
        train_ds = data_loader.IcebergDataset(
            X_train_fold,
            inc_train_fold,
            y_train_fold,
            transform=data_loader.get_transforms(augment=True),
            scaling_stats=scaling_stats,
        )
        val_ds = data_loader.IcebergDataset(
            X_val_fold,
            inc_val_fold,
            y_val_fold,
            transform=data_loader.get_transforms(augment=False),
            scaling_stats=scaling_stats,
        )

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Run Fold Training
        run_fold(fold_idx, train_loader, val_loader, device)

        # Inference on Test Set with Best Model of this Fold
        logger.info(f"Generating predictions for Fold {fold_idx}...")
        net = model.WBDIN().to(device)
        model_path = os.path.join(config.WORKING_DIR, f"model_fold_{fold_idx}.pth")
        net.load_state_dict(torch.load(model_path))

        fold_preds = predict(net, test_loader, device)
        test_predictions_sum += fold_preds

    # 5. Aggregate Predictions and Submit
    avg_preds = test_predictions_sum / config.N_FOLDS

    # Load sample submission or test metadata to get IDs
    df_test = pd.read_csv(os.path.join(config.METADATA_DIR, "test.csv"))
    if config.DEBUG:
        df_test = df_test.iloc[: len(X_test)]
    df_test["is_iceberg"] = avg_preds

    # Save submission
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_FILE), exist_ok=True)
    df_test[["id", "is_iceberg"]].to_csv(config.SUBMISSION_FILE, index=False)

    logger.info(f"Training complete. Submission saved to {config.SUBMISSION_FILE}")
