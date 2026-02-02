import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.model import SimpleFCN


class Trainer:
    """
    Trainer class to encapsulate the training, validation, and prediction logic
    for the Simple Fully Connected Network (SFCN).
    """

    def __init__(self, model=None):
        """
        Initialize the Trainer with model, criterion, optimizer, and device.
        """
        self.device = torch.device(Config.DEVICE)

        # Initialize model if not provided
        if model is not None:
            self.model = model
        else:
            self.model = SimpleFCN()

        self.model.to(self.device)

        # Objective function and Optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Early Stopping State
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.best_model_state = None

    def train_epoch(self, train_loader):
        """
        Executes one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = len(train_loader.dataset)

        for batch_x, batch_angle, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_angle = batch_angle.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch_x, batch_angle)
            loss = self.criterion(outputs, batch_y)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = len(val_loader.dataset)

        with torch.no_grad():
            for batch_x, batch_angle, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_angle = batch_angle.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x, batch_angle)
                loss = self.criterion(outputs, batch_y)

                running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
    ):
        """
        Runs the full training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Print metrics without rounding
            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
            )

            # Early Stopping Check
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_state = self.model.state_dict()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Load the best model state before finishing
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with Val Loss: {self.best_loss}")

    def predict(self, test_loader, test_ids):
        """
        Generates predictions for the test set and saves them to the submission file.
        """
        self.model.eval()
        predictions = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch_x, batch_angle in test_loader:
                batch_x = batch_x.to(self.device)
                batch_angle = batch_angle.to(self.device)

                outputs = self.model(batch_x, batch_angle)
                # Flatten outputs (Batch, 1) -> (Batch,)
                preds = outputs.cpu().numpy().flatten()
                predictions.extend(preds)

        # Create submission DataFrame
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": predictions})

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
