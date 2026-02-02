import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score

from library.config import Config
from library.model import CRWBN
from library.data_loader import process_and_cache_data, get_global_stats, IcebergDataset
from library.utils import setup_logger, set_seed


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0


class Trainer:
    def __init__(
        self, model, device, criterion, optimizer, scheduler=None, logger=None
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        for inputs, angles, targets in train_loader:
            inputs = inputs.to(self.device)
            angles = angles.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            outputs = self.model(inputs, angles)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # For metrics
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Clip preds for log_loss stability
        all_preds_clipped = np.clip(all_preds, 1e-15, 1 - 1e-15)
        epoch_log_loss = log_loss(all_targets, all_preds_clipped)

        # Accuracy (threshold 0.5)
        epoch_acc = accuracy_score(all_targets, (all_preds >= 0.5).astype(int))

        return epoch_loss, epoch_log_loss, epoch_acc

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for inputs, angles, targets in val_loader:
                inputs = inputs.to(self.device)
                angles = angles.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                outputs = self.model(inputs, angles)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(val_loader.dataset)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        all_preds_clipped = np.clip(all_preds, 1e-15, 1 - 1e-15)
        epoch_log_loss = log_loss(all_targets, all_preds_clipped)
        epoch_acc = accuracy_score(all_targets, (all_preds >= 0.5).astype(int))

        return epoch_loss, epoch_log_loss, epoch_acc

    def fit(self, train_loader, val_loader, epochs, patience, fold_idx):
        early_stopping = EarlyStopping(patience=patience, verbose=False)
        best_val_loss = float("inf")

        for epoch in range(epochs):
            train_loss, train_ll, train_acc = self.train_epoch(train_loader)
            val_loss, val_ll, val_acc = self.validate(val_loader)

            if self.scheduler:
                self.scheduler.step(val_loss)

            log_msg = (
                f"Fold {fold_idx} Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.6f} (LL: {train_ll:.6f}, Acc: {train_acc:.4f}) - "
                f"Val Loss: {val_loss:.6f} (LL: {val_ll:.6f}, Acc: {val_acc:.4f})"
            )

            if self.logger:
                self.logger.info(log_msg)
            else:
                print(log_msg)

            early_stopping(val_loss, self.model)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if early_stopping.early_stop:
                stop_msg = f"Early stopping at epoch {epoch+1}"
                if self.logger:
                    self.logger.info(stop_msg)
                else:
                    print(stop_msg)
                break

        # Load best weights
        if early_stopping.best_model_state:
            self.model.load_state_dict(early_stopping.best_model_state)

        return best_val_loss


def run_cv_training(debug=False):
    logger = setup_logger()
    set_seed(Config.SEED)

    # Load Data
    data = process_and_cache_data(load_cached_data=True)
    X_full = data["X_train_full"]
    y_full = data["y_train_full"]
    angles_full = data["angles_train_full"]
    ids_full = data["ids_train_full"]

    # Global Stats
    stats = get_global_stats(X_full)

    if debug:
        logger.info(f"Debug mode: Reducing dataset size to {Config.DEBUG_SIZE}")
        X_full = X_full[: Config.DEBUG_SIZE]
        y_full = y_full[: Config.DEBUG_SIZE]
        angles_full = angles_full[: Config.DEBUG_SIZE]
        ids_full = ids_full[: Config.DEBUG_SIZE]

    # K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        logger.info(f"Starting Fold {fold}")

        # Split Data
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]
        angles_train, angles_val = angles_full[train_idx], angles_full[val_idx]
        ids_train, ids_val = ids_full[train_idx], ids_full[val_idx]

        # Datasets
        train_ds = IcebergDataset(
            X_train, angles_train, y_train, ids_train, transform=True, stats=stats
        )
        val_ds = IcebergDataset(
            X_val, angles_val, y_val, ids_val, transform=False, stats=stats
        )

        # Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Model
        model = CRWBN().to(device)

        # Optimizer & Scheduler
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Trainer
        trainer = Trainer(model, device, criterion, optimizer, scheduler, logger)

        # Fit
        best_loss = trainer.fit(
            train_loader, val_loader, Config.EPOCHS, Config.PATIENCE, fold
        )
        fold_results.append(best_loss)

        # Save Model
        save_path = Config.MODEL_PATH_TEMPLATE.format(fold)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Saved model for Fold {fold} to {save_path}")

    logger.info(
        f"CV Training Completed. Average Best Val Loss: {np.mean(fold_results):.6f}"
    )


def generate_submission(debug=False):
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data
    data = process_and_cache_data(load_cached_data=True)
    X_test = data["X_test"]
    angles_test = data["angles_test"]
    ids_test = data["ids_test"]

    # We need global stats from training data for consistent normalization
    X_train_full = data["X_train_full"]
    stats = get_global_stats(X_train_full)

    if debug:
        X_test = X_test[: Config.DEBUG_SIZE]
        angles_test = angles_test[: Config.DEBUG_SIZE]
        ids_test = ids_test[: Config.DEBUG_SIZE]

    test_ds = IcebergDataset(
        X_test, angles_test, labels=None, ids=ids_test, transform=False, stats=stats
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Ensemble Predictions
    fold_preds = []

    for fold in range(Config.NUM_FOLDS):
        model_path = Config.MODEL_PATH_TEMPLATE.format(fold)
        if not os.path.exists(model_path):
            logger.warning(
                f"Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        logger.info(f"Loading model for Fold {fold}...")
        model = CRWBN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        preds = []
        with torch.no_grad():
            for inputs, angles, _ in test_loader:
                inputs = inputs.to(device)
                angles = angles.to(device)

                outputs = model(inputs, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds.extend(probs)

        fold_preds.append(np.array(preds))

    if not fold_preds:
        logger.error("No models loaded. Cannot generate submission.")
        return

    # Average predictions
    avg_preds = np.mean(fold_preds, axis=0)

    # Create DataFrame
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
