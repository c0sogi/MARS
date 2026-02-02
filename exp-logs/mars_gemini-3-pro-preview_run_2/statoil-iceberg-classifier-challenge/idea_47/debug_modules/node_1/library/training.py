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
from library.utils import seed_everything
from library.model import PCWBN
from library.data_handling import load_data, IcebergDataset, get_global_stats


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs. Saves the best model state.
    """

    def __init__(self, patience=10, min_delta=0, path="checkpoint.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        self.best_model_state = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), self.path)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, inc_angles, labels in loader:
        inputs = inputs.to(device)
        inc_angles = inc_angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        outputs = model(inputs, inc_angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Accuracy calculation
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, inc_angles, labels in loader:
            inputs = inputs.to(device)
            inc_angles = inc_angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs, inc_angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_k_fold(debug_size=None, epochs=Config.NUM_EPOCHS):
    """
    Executes Stratified K-Fold Cross Validation.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 1. Load Data
    data = load_data(load_cached_data=True)

    # Merge Train and Val splits to perform full K-Fold
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    inc_full = np.concatenate([data["inc_train"], data["inc_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)

    if debug_size is not None:
        print(f"DEBUG: Truncating data to {debug_size} samples")
        X_full = X_full[:debug_size]
        inc_full = inc_full[:debug_size]
        y_full = y_full[:debug_size]

    # Re-compute global stats on the full dataset for consistency
    # This aligns with the idea: "statistics derived from the entire training dataset"
    full_stats = get_global_stats(X_full, inc_full)

    # 2. K-Fold Setup
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n{'='*20} Fold {fold+1}/{Config.NUM_FOLDS} {'='*20}")

        # Split Data
        X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
        inc_train_fold, inc_val_fold = inc_full[train_idx], inc_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(
            X_train_fold, inc_train_fold, y_train_fold, full_stats, transform=True
        )
        val_ds = IcebergDataset(
            X_val_fold, inc_val_fold, y_val_fold, full_stats, transform=False
        )

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

        # Initialize Model
        model = PCWBN().to(device)

        # Optimization
        criterion = nn.BCEWithLogitsLoss()
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

        # Early Stopping
        model_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold}.pth")
        early_stopping = EarlyStopping(patience=Config.PATIENCE, path=model_path)

        # Training Loop
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.10f}, Train Acc: {train_acc:.10f}, "
                f"Val Loss: {val_loss:.10f}, Val Acc: {val_acc:.10f}"
            )

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # Load best weights
        model.load_state_dict(early_stopping.best_model_state)
        fold_results.append(early_stopping.best_loss)

        # Clean up
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    print(
        f"\nCross-Validation Complete. Average Best Loss: {np.mean(fold_results):.10f}"
    )
    return full_stats


def predict_and_submit(stats, debug_size=None):
    """
    Generates predictions using the ensemble of trained models and saves submission.
    """
    print("\nGenerating Submission...")
    device = Config.DEVICE

    # 1. Load Test Data
    data = load_data(load_cached_data=True)
    X_test = data["X_test"]
    inc_test = data["inc_test"]

    if debug_size is not None:
        X_test = X_test[:debug_size]
        inc_test = inc_test[:debug_size]

    # Dummy labels for dataset
    y_test_dummy = np.zeros(len(X_test))

    test_ds = IcebergDataset(X_test, inc_test, y_test_dummy, stats, transform=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Ensemble Prediction
    ensemble_preds = np.zeros((len(X_test), 1))

    for fold in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        print(f"Predicting with model fold {fold}...")
        model = PCWBN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for inputs, inc_angles, _ in test_loader:
                inputs = inputs.to(device)
                inc_angles = inc_angles.to(device)

                outputs = model(inputs, inc_angles)
                probs = torch.sigmoid(outputs)
                fold_preds.append(probs.cpu().numpy())

        fold_preds = np.concatenate(fold_preds, axis=0)
        ensemble_preds += fold_preds

    # Average predictions
    avg_preds = ensemble_preds / Config.NUM_FOLDS

    # 3. Create Submission File
    # Load test IDs from metadata
    test_meta = pd.read_csv(Config.TEST_META_PATH)
    if debug_size is not None:
        test_meta = test_meta.iloc[:debug_size]

    submission = pd.DataFrame(
        {"id": test_meta["id"], "is_iceberg": avg_preds.flatten()}
    )

    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run_training_pipeline(debug=False):
    """
    Main entry point to run the full training and submission pipeline.
    """
    debug_size = 100 if debug else None
    epochs = 2 if debug else Config.NUM_EPOCHS

    # Train 5 folds
    stats = train_k_fold(debug_size=debug_size, epochs=epochs)

    # Generate submission
    predict_and_submit(stats, debug_size=debug_size)
