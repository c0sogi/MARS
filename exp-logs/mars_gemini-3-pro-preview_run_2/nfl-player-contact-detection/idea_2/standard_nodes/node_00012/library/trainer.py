import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    POS_WEIGHT,
    MODEL_CHECKPOINT_PATH,
    SEED,
    WORKING_DIR,
)
from library.utils import seed_everything, compute_mcc, optimize_threshold
from library.model import KinematicMLP
from library.dataset import NFLSequenceDataset


class Trainer:
    """
    Manages the training, validation, and inference of the KinematicMLP model.
    """

    def __init__(self):
        seed_everything(SEED)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer initialized on device: {self.device}")

        # Initialize Model
        self.model = KinematicMLP().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)

        # Initialize Loss Function with Class Weighting
        # POS_WEIGHT handles the ~1:72 imbalance
        pos_weight = torch.tensor([POS_WEIGHT], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            # Unpack batch
            features = batch["features"].to(self.device)
            is_ground = batch["is_ground"].to(self.device)
            targets = batch["target"].to(self.device).unsqueeze(1)  # (Batch, 1)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(features, is_ground)

            # Loss calculation
            loss = self.criterion(logits, targets)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns average loss, predicted probabilities, and true labels.
        """
        self.model.eval()
        running_loss = 0.0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                is_ground = batch["is_ground"].to(self.device)
                targets = batch["target"].to(self.device).unsqueeze(1)

                logits = self.model(features, is_ground)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * features.size(0)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        epoch_loss = running_loss / len(val_loader.dataset)
        all_probs = np.concatenate(all_probs).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        return epoch_loss, all_probs, all_targets

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Main training loop with Early Stopping based on Validation MCC.
        """
        # Create Datasets and Loaders
        train_dataset = NFLSequenceDataset(X_train, y_train)
        val_dataset = NFLSequenceDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        best_val_mcc = -1.0
        patience_counter = 0

        print("Starting training...")

        for epoch in range(1, EPOCHS + 1):
            # Training Step
            train_loss = self.train_epoch(train_loader)

            # Validation Step
            val_loss, val_probs, val_targets = self.validate(val_loader)

            # Optimize Threshold for MCC
            # We optimize threshold every epoch to accurately gauge performance potential
            best_thresh, val_mcc = optimize_threshold(val_targets, val_probs)

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MCC: {val_mcc} | "
                f"Best Thresh: {best_thresh:.4f}"
            )

            # Early Stopping Logic
            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                patience_counter = 0

                # Save best model
                torch.save(self.model.state_dict(), MODEL_CHECKPOINT_PATH)
                print(f"  -> New best model saved to {MODEL_CHECKPOINT_PATH}")
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation MCC: {best_val_mcc}")

        # Load best model state for future inference
        self.model.load_state_dict(
            torch.load(MODEL_CHECKPOINT_PATH, map_location=self.device)
        )

    def predict(self, X_test):
        """
        Generates probability predictions for the test set.
        """
        dataset = NFLSequenceDataset(X_test, y=None)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        self.model.eval()
        all_probs = []

        # Load best model if available
        if os.path.exists(MODEL_CHECKPOINT_PATH):
            try:
                self.model.load_state_dict(
                    torch.load(MODEL_CHECKPOINT_PATH, map_location=self.device)
                )
            except Exception as e:
                print(
                    f"Warning: Could not load checkpoint, using current weights. Error: {e}"
                )

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                is_ground = batch["is_ground"].to(self.device)

                logits = self.model(features, is_ground)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs).flatten()
