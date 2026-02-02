import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library import config, data, model


class Trainer:
    """
    Trainer class to manage the training and evaluation of the Stabilized 2.5D Net.
    """

    def __init__(self, learning_rate=config.LEARNING_RATE, device=config.DEVICE):
        self.device = device
        self.learning_rate = learning_rate

        # Ensure reproducibility
        model.set_seed(config.SEED)

        # Initialize Model
        self.model = model.Stabilized25DNet().to(self.device)

        # Initialize Optimizer and Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

        # Early Stopping State
        self.best_val_auc = 0.0
        self.patience_counter = 0

    def train(self, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE):
        """
        Executes the training loop with early stopping.
        """
        # Get DataLoaders (Data processing and caching is handled within data.py)
        train_loader, val_loader, _, _ = data.get_dataloaders(batch_size=batch_size)

        print(f"Starting training on {self.device} for {epochs} epochs...")

        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_auc = model.train_epoch(
                self.model, train_loader, self.criterion, self.optimizer, self.device
            )

            # Validate
            val_loss, val_auc = model.validate(
                self.model, val_loader, self.criterion, self.device
            )

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Train AUC: {train_auc} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Early Stopping & Checkpointing
            if val_auc > self.best_val_auc + config.MIN_DELTA:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                # Save the best model
                torch.save(self.model.state_dict(), config.MODEL_SAVE_PATH)
            else:
                self.patience_counter += 1

            if self.patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Training complete. Best Val AUC: {self.best_val_auc}")

    def predict(self, batch_size=config.BATCH_SIZE):
        """
        Loads the best model and generates predictions for the test set.
        """
        # Load best model state
        if os.path.exists(config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            print(
                "Warning: No trained model found. Predictions will be random/untrained."
            )

        self.model.eval()

        # Get Test DataLoader
        _, _, test_loader, test_ids = data.get_dataloaders(batch_size=batch_size)

        predictions = []
        print("Generating predictions...")

        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                predictions.extend(probs)

        # Save Submission
        submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": predictions})

        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
