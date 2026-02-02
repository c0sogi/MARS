import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

from library.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    CHECKPOINT_DIR,
    N_FOLDS,
    SUBMISSION_DIR,
    process_data,
    IcebergDataset,
)
from library.model import IDPH_CNN
from library.utils import seed_everything, get_device


class Trainer:
    """
    Manages the training lifecycle for a single model fold.
    """

    def __init__(self, model, device, fold_idx):
        self.model = model
        self.device = device
        self.fold_idx = fold_idx
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer: AdamW with constant learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
        )

        self.best_loss = float("inf")
        self.patience_counter = 0
        self.checkpoint_path = os.path.join(
            CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth"
        )

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for imgs, angs, lbls in train_loader:
            imgs = imgs.to(self.device)
            angs = angs.to(self.device)
            lbls = lbls.to(self.device)

            batch_size = imgs.size(0)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(imgs, angs).squeeze(1)
            loss = self.criterion(outputs, lbls)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        with torch.no_grad():
            for imgs, angs, lbls in val_loader:
                imgs = imgs.to(self.device)
                angs = angs.to(self.device)
                lbls = lbls.to(self.device)

                batch_size = imgs.size(0)

                outputs = self.model(imgs, angs).squeeze(1)
                loss = self.criterion(outputs, lbls)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
        return epoch_loss

    def fit(self, train_loader, val_loader, epochs=NUM_EPOCHS):
        print(f"Starting training for Fold {self.fold_idx + 1}")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Print full precision as requested
            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        return self.best_loss


def run_cross_validation(load_cached_data=True):
    """
    Orchestrates the 5-Fold Cross-Validation training pipeline.
    """
    device = get_device()
    print(f"Device: {device}")

    # Load data using the cached processor
    # X_train here contains the full labeled dataset (train.csv + val.csv)
    data = process_data(load_cached_data=load_cached_data)
    X_full, y_full, angle_full, ids_full, _, _, _ = data

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    cv_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")

        # Split data
        X_tr, X_val = X_full[train_idx], X_full[val_idx]
        y_tr, y_val = y_full[train_idx], y_full[val_idx]
        a_tr, a_val = angle_full[train_idx], angle_full[val_idx]

        # Transforms
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        )

        # Datasets
        train_ds = IcebergDataset(X_tr, y_tr, a_tr, transform=train_transform)
        val_ds = IcebergDataset(X_val, y_val, a_val, transform=None)

        # Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )

        # Model
        model = IDPH_CNN().to(device)

        # Trainer
        trainer = Trainer(model, device, fold_idx)
        best_loss = trainer.fit(train_loader, val_loader)
        cv_scores.append(best_loss)

    print(f"\nCV Log Loss: {np.mean(cv_scores)} (+/- {np.std(cv_scores)})")
    return cv_scores


def generate_submission(load_cached_data=True):
    """
    Generates predictions for the test set using the trained models from all folds.
    """
    device = get_device()

    # Load test data
    data = process_data(load_cached_data=load_cached_data)
    _, _, _, _, X_test, angle_test, ids_test = data

    test_ds = IcebergDataset(X_test, None, angle_test, transform=None)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    fold_preds = []

    print("\nGenerating predictions...")

    for fold_idx in range(N_FOLDS):
        model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Checkpoint for fold {fold_idx} not found at {model_path}")
            continue

        model = IDPH_CNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        preds = []
        with torch.no_grad():
            for imgs, angs in test_loader:
                imgs = imgs.to(device)
                angs = angs.to(device)

                outputs = model(imgs, angs).squeeze(1)
                probs = torch.sigmoid(outputs)
                preds.extend(probs.cpu().numpy())

        fold_preds.append(np.array(preds))

    if not fold_preds:
        raise RuntimeError("No predictions generated. Check if models are trained.")

    # Average predictions
    avg_preds = np.mean(fold_preds, axis=0)

    # Save submission
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
