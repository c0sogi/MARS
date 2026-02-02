import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    MIN_LR,
    SEED,
)
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import EWADeepSets


class Trainer:
    """
    Manages the training, validation, and prediction process for the EWADeepSets model.
    """

    def __init__(self, model):
        self.model = model.to(DEVICE)
        # Target variables are already log-transformed in the dataset.
        # MSE on log-transformed data is equivalent to MSLE on original data.
        self.criterion = nn.MSELoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=MIN_LR,
        )

        self.best_val_loss = float("inf")

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Unpack batch and move to device
            atomic_feats = batch["atomic_features"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            global_feats = batch["global_features"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(atomic_feats, mask, global_feats)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Accumulate loss (MSE * batch_size)
            total_loss += loss.item() * targets.size(0)

        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                atomic_feats = batch["atomic_features"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                global_feats = batch["global_features"].to(DEVICE)
                targets = batch["targets"].to(DEVICE)

                outputs = self.model(atomic_feats, mask, global_feats)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * targets.size(0)

        avg_loss = total_loss / len(val_loader.dataset)
        # RMSLE corresponds to sqrt of MSE on log-transformed targets
        rmsle = np.sqrt(avg_loss)
        return avg_loss, rmsle

    def fit(self, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE):
        print(f"Starting training on device: {DEVICE}")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_rmsle = self.validate(val_loader)

            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val RMSLE: {val_rmsle}"
            )

            # Early Stopping Check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                save_path = os.path.join(WORKING_DIR, "best_model.pt")
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

    def predict(self, test_loader):
        # Load best model weights
        model_path = os.path.join(WORKING_DIR, "best_model.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Loaded best model from {model_path}")
        else:
            print("Warning: No best model found, using current state.")

        self.model.eval()
        all_preds = []
        all_ids = []

        with torch.no_grad():
            for batch in test_loader:
                atomic_feats = batch["atomic_features"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                global_feats = batch["global_features"].to(DEVICE)
                ids = batch["ids"]

                outputs = self.model(atomic_feats, mask, global_feats)

                # Move to CPU
                preds = outputs.cpu().numpy()

                # Inverse transform: exp(x) - 1 to get back to original scale
                # (Since targets were log1p transformed)
                preds_original = np.expm1(preds)

                all_preds.append(preds_original)
                all_ids.append(ids.numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_ids = np.concatenate(all_ids, axis=0)

        return all_ids, all_preds


def run_training(load_cached_data=True, epochs=EPOCHS, patience=PATIENCE):
    """
    Orchestrates the training pipeline:
    1. Sets seeds.
    2. Loads data.
    3. Initializes model and trainer.
    4. Runs training loop.
    5. Generates and saves submission.
    """
    set_seed(SEED)

    # 1. Get DataLoaders (caching handled within this function)
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    model = EWADeepSets()

    # 3. Initialize Trainer
    trainer = Trainer(model)

    # 4. Train Model
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=patience)

    # 5. Generate Predictions
    ids, preds = trainer.predict(test_loader)

    # 6. Create Submission File
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds[:, 0],
            "bandgap_energy_ev": preds[:, 1],
        }
    )

    # Sort by ID
    submission_df = submission_df.sort_values("id")

    # Save to disk
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission_df.head())
