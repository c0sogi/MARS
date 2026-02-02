import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, compute_roc_auc, get_device
from library.model import S3HDNetwork
from library.data_loader import get_dataloaders


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the S3HD Network.
    """

    def __init__(self):
        self.device = get_device()
        self.model = S3HDNetwork().to(self.device)
        # BCEWithLogitsLoss is more numerically stable than Sigmoid + BCELoss
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.best_val_auc = 0.0

    def train_epoch(self, train_loader):
        """
        Executes one training epoch.
        """
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_probs = []

        for inputs, targets, _ in train_loader:
            inputs = inputs.to(self.device)
            # Ensure targets have shape (B, 1) to match logits
            targets = targets.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to convert logits to probabilities for AUC
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_auc = compute_roc_auc(all_targets, all_probs)
        return epoch_loss, epoch_auc

    def validate(self, val_loader):
        """
        Evaluates the model on the validation dataset.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_auc = compute_roc_auc(all_targets, all_probs)
        return epoch_loss, epoch_auc

    def fit(self, train_loader, val_loader, epochs=Config.NUM_EPOCHS, patience=5):
        """
        Runs the training loop with Early Stopping and Model Checkpointing.
        """
        print(f"Starting training on device: {self.device}")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_auc = self.train_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss}, Train AUC: {train_auc}, "
                f"Val Loss: {val_loss}, Val AUC: {val_auc}"
            )

            # Checkpoint: Save best model based on Validation AUC
            if val_auc > self.best_val_auc:
                print(
                    f"Validation AUC improved from {self.best_val_auc} to {val_auc}. Saving model..."
                )
                self.best_val_auc = val_auc
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print(f"Training complete. Best Validation AUC: {self.best_val_auc}")

    def predict(self, test_loader, output_path=Config.SUBMISSION_PATH):
        """
        Generates predictions for the test set and saves to CSV.
        """
        print("Loading best model for inference...")
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            print("Warning: Best model not found. Using current model state.")

        self.model.eval()
        predictions = []
        ids = []

        print("Generating predictions...")
        with torch.no_grad():
            for inputs, _, batch_ids in test_loader:
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                predictions.extend(probs)
                ids.extend(batch_ids)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save submission
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")


def run_training_pipeline(load_cached_data=True):
    """
    Main entry point for the training and submission pipeline.
    """
    # 1. Set Seed for Reproducibility
    set_seed(Config.SEED)

    # 2. Get DataLoaders (handles caching internally)
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 3. Initialize Trainer
    trainer = Trainer()

    # 4. Train Model
    trainer.fit(train_loader, val_loader)

    # 5. Generate Submission
    trainer.predict(test_loader)
