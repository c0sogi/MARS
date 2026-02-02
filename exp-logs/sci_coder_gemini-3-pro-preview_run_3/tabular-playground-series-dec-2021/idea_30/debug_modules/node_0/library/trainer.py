import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SCHEDULER_MODE,
    PATIENCE,
    WORKING_DIR,
    MODEL_PATH,
    SUBMISSION_PATH,
    DEVICE,
    HIDDEN_DIM,
    NUM_CLASSES,
    DROPOUT,
)
from library.utils import seed_everything
from library.model import ParallelDCNResNet
from library.data_loader import get_dataloaders


class Trainer:
    """
    Trainer class to handle model training, validation, and early stopping.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

        # AdamW Optimizer (Decoupled Weight Decay)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # ReduceLROnPlateau Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=SCHEDULER_MODE,
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            verbose=True,
        )

    def train_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)

                running_loss += loss.item() * X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self, epochs=EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        best_val_acc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss}, Train Acc: {train_acc}, "
                f"Val Loss: {val_loss}, Val Acc: {val_acc}"
            )

            # Step scheduler based on validation accuracy
            self.scheduler.step(val_acc)

            # Early Stopping Logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Restoring best model with Val Acc: {best_val_acc}")
        self.model.load_state_dict(best_model_wts)

        # Save best model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")


def generate_predictions(model, test_loader, test_ids, device):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for X in test_loader:
            X = X.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            # Convert 0-indexed predictions back to 1-indexed labels (1-7)
            predictions.extend((predicted + 1).cpu().numpy())

    # Create DataFrame
    df_sub = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    # Save submission
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


def run_training(epochs=EPOCHS):
    """
    Orchestrates the full training pipeline.
    """
    seed_everything()

    # Get Dataloaders
    # load_cached_data=True ensures we use the cache if available, or create it
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # Infer Input Dimension from a single batch
    # We grab one batch from train_loader to check shape
    sample_X, _ = next(iter(train_loader))
    input_dim = sample_X.shape[1]

    # Initialize Model
    model = ParallelDCNResNet(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    )
    model.to(DEVICE)

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, DEVICE)

    # Train
    trainer.fit(epochs=epochs)

    # Predict
    generate_predictions(trainer.model, test_loader, test_ids, DEVICE)
