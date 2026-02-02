import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything
from library.model import DMWBNet
from library.data import process_data, make_dataloaders, IcebergDataset
from torch.utils.data import DataLoader


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model weights using deepcopy.
    """

    def __init__(self, patience=Config.PATIENCE, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Trainer:
    """
    Manages the training and validation loop for the model.
    """

    def __init__(self, model, device, criterion, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for inputs, angles, labels in train_loader:
            inputs = inputs.to(self.device)
            angles = angles.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs, angles)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, angles, labels in val_loader:
                inputs = inputs.to(self.device)
                angles = angles.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs, angles)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        return epoch_loss

    def fit(self, train_loader, val_loader, epochs, early_stopping):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            if self.scheduler:
                self.scheduler.step(val_loss)

            # Print full precision as requested
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            early_stopping(val_loss, self.model)

            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best weights
        if early_stopping.best_model_wts is not None:
            self.model.load_state_dict(early_stopping.best_model_wts)

        return early_stopping.best_loss


def run_kfold_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Executes the Stratified K-Fold training pipeline.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and process data
    # process_data handles caching internally
    X, y, inc, X_test, inc_test, test_ids = process_data(load_cached_data=True)

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Array to accumulate test predictions
    test_preds_accumulator = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold + 1}/{Config.N_FOLDS} ===")

        # Create DataLoaders
        train_loader, val_loader = make_dataloaders(
            X, y, inc, train_idx, val_idx, batch_size=batch_size
        )

        # Initialize Model and Components
        model = DMWBNet().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        early_stopping = EarlyStopping(patience=Config.PATIENCE)
        trainer = Trainer(model, device, criterion, optimizer, scheduler)

        # Train
        trainer.fit(train_loader, val_loader, epochs, early_stopping)

        # Save Best Model
        model_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold}.pth")
        torch.save(early_stopping.best_model_wts, model_path)
        print(f"Saved best model for fold {fold} to {model_path}")

        # Inference on Test Set
        model.eval()
        test_ds = IcebergDataset(X_test, inc_test, transform=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        fold_preds = []
        with torch.no_grad():
            for inputs, angles in test_loader:
                inputs = inputs.to(device)
                angles = angles.to(device)
                outputs = model(inputs, angles)
                fold_preds.extend(outputs.cpu().numpy().flatten())

        test_preds_accumulator += np.array(fold_preds)

    # Average Predictions
    final_preds = test_preds_accumulator / Config.N_FOLDS

    # Generate Submission
    submission = pd.DataFrame({"id": test_ids, "is_iceberg": final_preds})
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
