import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, calculate_global_stats
from library.data_loader import process_data, IcebergDataset, get_transforms
from library.model import RIWBN


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                # Print is handled in the main loop via epoch stats, keeping this silent to avoid clutter
                pass
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        # IcebergDataset returns: img_tensor, inc_tensor, label_tensor (if labels exist)
        inputs, inc, targets = batch

        inputs = inputs.to(device)
        inc = inc.to(device)
        targets = targets.to(device).view(
            -1, 1
        )  # Ensure target shape matches logit shape

        optimizer.zero_grad()

        outputs = model(inputs, inc)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            inputs, inc, targets = batch

            inputs = inputs.to(device)
            inc = inc.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(inputs, inc)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def predict(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # Test loader might not return labels
            if len(batch) == 3:
                inputs, inc, _ = batch
            else:
                inputs, inc = batch

            inputs = inputs.to(device)
            inc = inc.to(device)

            outputs = model(inputs, inc)
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy().flatten())

    return np.array(all_preds)


def run_training():
    seed_everything(Config.SEED)

    # 1. Prepare Data
    # Load cached data
    data = process_data(load_cached_data=True)
    stats = calculate_global_stats(load_cached_data=True, debug=Config.DEBUG)

    # Combine Train and Val for Stratified K-Fold CV
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    inc_full = np.concatenate([data["inc_train"], data["inc_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)

    # Calculate B3 (Mean Channel) stats using the full training set
    # This is required because calculate_global_stats only handles B1 and B2
    b3_data = X_full[:, 2, :, :]
    stats["b3_min"] = float(b3_data.min())
    stats["b3_max"] = float(b3_data.max())

    # Prepare Test Data
    test_dataset = IcebergDataset(
        X=data["X_test"],
        inc_angles=data["inc_test"],
        labels=None,
        transform=get_transforms("test"),
        global_stats=stats,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # 2. Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_preds = np.zeros((len(test_dataset),))

    print(f"Starting {Config.NUM_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n=== Fold {fold + 1}/{Config.NUM_FOLDS} ===")

        # Split Data
        X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
        inc_train_fold, inc_val_fold = inc_full[train_idx], inc_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(
            X=X_train_fold,
            inc_angles=inc_train_fold,
            labels=y_train_fold,
            transform=get_transforms("train"),
            global_stats=stats,
        )
        val_ds = IcebergDataset(
            X=X_val_fold,
            inc_angles=inc_val_fold,
            labels=y_val_fold,
            transform=get_transforms("val"),
            global_stats=stats,
        )

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

        # Initialize Model
        model = RIWBN().to(Config.DEVICE)

        # Optimizer & Scheduler & Loss
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Early Stopping
        early_stopping = EarlyStopping(patience=Config.PATIENCE, verbose=True)

        # Training Loop
        for epoch in range(Config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, Config.DEVICE
            )
            val_loss = validate(model, val_loader, criterion, Config.DEVICE)

            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss:.10f} - "
                f"Val Loss: {val_loss:.10f}"
            )

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # Load Best Weights
        if early_stopping.best_state:
            model.load_state_dict(early_stopping.best_state)

        # Save Model
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved best model for fold {fold+1} to {model_path}")

        # Predict on Test
        print("Generating predictions for fold...")
        preds = predict(model, test_loader, Config.DEVICE)
        fold_preds += preds

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()

    # Average Predictions
    avg_preds = fold_preds / Config.NUM_FOLDS

    # Create Submission
    submission = pd.DataFrame({"id": data["ids_test"], "is_iceberg": avg_preds})

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
