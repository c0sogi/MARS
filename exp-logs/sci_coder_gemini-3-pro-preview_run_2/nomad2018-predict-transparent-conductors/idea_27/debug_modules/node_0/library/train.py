import time
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from library.config import Config
from library.model import SPRACGN
from library.data import get_dataloaders
from library.utils import set_seed, StandardScaler, compute_metric


class Trainer:
    """
    Manages the training, validation, and prediction processes for the SP-RA-CGN model.
    """

    def __init__(self, model, device=Config.DEVICE):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.SCHEDULER_MIN_LR,
        )
        self.scaler = StandardScaler(device=device)
        self.criterion = nn.MSELoss()

    def fit_scaler(self, train_loader):
        """
        Computes mean and std of targets from the training set and saves the scaler.
        """
        all_y = []
        for batch in train_loader:
            if batch.y is not None:
                all_y.append(batch.y)

        if all_y:
            y_tensor = torch.cat(all_y, dim=0)
            self.scaler.fit(y_tensor)
            print("Target scaler fitted on training data.")
            self.scaler.save(Config.TARGET_SCALER_PATH)
        else:
            print("Warning: No targets found in training loader to fit scaler.")

    def train_epoch(self, train_loader):
        """
        Runs one training epoch.
        """
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(batch)

            # Standardize targets
            target = batch.y
            target_scaled = self.scaler.transform(target)

            # Compute loss
            loss = self.criterion(pred, target_scaled)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Record loss (MSE on scaled data)
            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs

        return total_loss / count if count > 0 else 0.0

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set using RMSLE.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                # Forward pass
                pred_scaled = self.model(batch)

                # Inverse transform predictions to original scale
                pred = self.scaler.inverse_transform(pred_scaled)

                all_preds.append(pred.cpu())
                all_targets.append(batch.y.cpu())

        if not all_preds:
            return 0.0

        y_pred = torch.cat(all_preds, dim=0)
        y_true = torch.cat(all_targets, dim=0)

        # Compute metric (RMSLE)
        score = compute_metric(y_true, y_pred)
        return score

    def fit(self, train_loader, val_loader, epochs=Config.NUM_EPOCHS):
        """
        Main training loop with early stopping and scheduling.
        """
        best_val_score = float("inf")
        patience_counter = 0

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_score = self.evaluate(val_loader)

            # Update scheduler based on validation score
            self.scheduler.step(val_score)

            # Checkpoint and Early Stopping
            if val_score < best_val_score:
                best_val_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            else:
                patience_counter += 1

            elapsed = time.time() - start_time
            # Printing full precision for validation score
            print(
                f"Epoch {epoch}: Train Loss = {train_loss}, Val RMSLE = {val_score}, Time = {elapsed}s"
            )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete. Best Val RMSLE: {best_val_score}")


def train_model(load_cached_data=True, dataset_size=None, epochs=Config.NUM_EPOCHS):
    """
    Initializes the environment, loads data, and trains the model.
    """
    set_seed(Config.SEED)
    Config.setup()

    # Load DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=load_cached_data, dataset_size=dataset_size
    )

    # Initialize Model
    model = SPRACGN()

    # Initialize Trainer
    trainer = Trainer(model)

    # Fit Scaler on Training Data
    trainer.fit_scaler(train_loader)

    # Start Training
    trainer.fit(train_loader, val_loader, epochs=epochs)


def generate_submission(load_cached_data=True, dataset_size=None):
    """
    Generates predictions for the test set using the best trained model.
    """
    set_seed(Config.SEED)

    # Load Test DataLoader
    # We use the same get_dataloaders function but only need the test_loader
    _, _, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, dataset_size=dataset_size
    )

    # Load Model
    model = SPRACGN()
    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}. Train the model first."
        )

    model.load_state_dict(
        torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=Config.DEVICE)
    )
    model.to(Config.DEVICE)
    model.eval()

    # Load Scaler
    scaler = StandardScaler(device=Config.DEVICE)
    if not os.path.exists(Config.TARGET_SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler state not found at {Config.TARGET_SCALER_PATH}. Train the model first."
        )
    scaler.load(Config.TARGET_SCALER_PATH)

    # Predict
    all_preds = []
    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(Config.DEVICE)
            pred_scaled = model(batch)
            pred = scaler.inverse_transform(pred_scaled)
            all_preds.append(pred.cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)

    # Prepare Submission DataFrame
    # Load test metadata to get IDs (preserving order)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    if dataset_size:
        test_meta = test_meta.iloc[:dataset_size]

    submission_df = pd.DataFrame(
        {
            "id": test_meta["id"],
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Save Submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
