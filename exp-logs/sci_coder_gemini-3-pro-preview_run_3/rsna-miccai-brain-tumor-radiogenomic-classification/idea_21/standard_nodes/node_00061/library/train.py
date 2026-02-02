import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import MGSHDNetwork


class Trainer:
    """
    Manages the training, validation, and inference process for the MG-SHD Network.
    """

    def __init__(self, train_loader, val_loader, test_loader):
        self.device = get_device()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Initialize Model
        self.model = MGSHDNetwork().to(self.device)

        # Optimizer and Loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=Config.LEARNING_RATE
        )
        self.criterion = nn.BCEWithLogitsLoss()

        # State tracking
        self.best_auc = 0.0
        self.best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    def train_one_epoch(self):
        """
        Performs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = len(self.train_loader.dataset)

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).float()

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self):
        """
        Evaluates the model on the validation set.
        Returns:
            val_loss (float): Average validation loss.
            val_auc (float): ROC AUC score.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []
        dataset_size = len(self.val_loader.dataset)

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).float()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

                # Apply sigmoid for probabilities
                probs = torch.sigmoid(outputs)

                all_targets.extend(targets.cpu().numpy().flatten())
                all_preds.extend(probs.cpu().numpy().flatten())

        val_loss = running_loss / dataset_size

        # Calculate AUC
        # Handle edge case where only one class is present
        try:
            if len(np.unique(all_targets)) > 1:
                val_auc = roc_auc_score(all_targets, all_preds)
            else:
                val_auc = 0.5
        except Exception:
            val_auc = 0.5

        return val_loss, val_auc

    def fit(self, epochs=Config.EPOCHS, patience=5):
        """
        Runs the full training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        no_improve_epochs = 0

        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_auc = self.validate()

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
            )

            # Checkpoint and Early Stopping
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved! AUC: {val_auc}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print(
                    f"Early stopping triggered. No improvement for {patience} epochs."
                )
                break

        print(f"Training finished. Best Validation AUC: {self.best_auc}")

    def generate_submission(self):
        """
        Generates predictions for the test set using the best model and saves to CSV.
        """
        print("Generating submission...")

        # Load best model weights
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print("Loaded best model weights.")
        else:
            print("Warning: Best model weights not found. Using current model state.")

        self.model.eval()
        predictions = []
        ids = []

        with torch.no_grad():
            for inputs, batch_ids in self.test_loader:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                predictions.extend(probs)
                ids.extend(batch_ids.numpy())

        # Create Submission DataFrame
        df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

        # Ensure BraTS21ID is formatted as 5-digit string (e.g., 00001)
        df["BraTS21ID"] = df["BraTS21ID"].apply(lambda x: f"{int(x):05d}")

        # Sort by ID
        df = df.sort_values("BraTS21ID")

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    """
    Main entry point to execute the training pipeline.
    """
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Data Loading
    # load_cached_data=True allows using pre-processed arrays if available
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Training
    trainer = Trainer(train_loader, val_loader, test_loader)
    trainer.fit(epochs=Config.EPOCHS, patience=5)

    # 4. Submission
    trainer.generate_submission()
