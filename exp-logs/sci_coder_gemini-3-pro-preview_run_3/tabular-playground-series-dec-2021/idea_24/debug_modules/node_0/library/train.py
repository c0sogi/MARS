import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, get_device
from library.model import ParallelDCNResNet
from library.data_loader import get_dataloaders


class Trainer:
    """
    Manages the training, validation, and optimization of the model.
    """

    def __init__(
        self,
        model,
        device,
        train_loader,
        val_loader,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        patience=Config.PATIENCE,
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.patience = patience

        # Optimizer: AdamW (Decoupled Weight Decay)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler: ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self):
        """
        Performs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        """
        Performs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, epochs=Config.EPOCHS):
        """
        Runs the full training loop with Early Stopping.
        """
        best_acc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Train Acc: {train_acc} | Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # Step the scheduler based on validation accuracy
            self.scheduler.step(val_acc)

            # Early Stopping Logic
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                # Save intermediate best model
                torch.save(
                    best_model_wts, os.path.join(Config.WORKING_DIR, "best_model.pth")
                )
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Training Complete. Best Validation Accuracy: {best_acc}")

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model


def train(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Main function to setup data, model, and run training.
    """
    seed_everything(Config.SEED)
    device = get_device()

    # Load Data
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # Determine input dimension from a sample batch
    X_sample, _ = next(iter(train_loader))
    input_dim = X_sample.shape[1]

    # Initialize Model
    model = ParallelDCNResNet(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_BLOCKS,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT,
    ).to(device)

    # Initialize Trainer
    trainer = Trainer(model, device, train_loader, val_loader)

    # Run Training
    best_model = trainer.fit(epochs=epochs)

    return best_model, test_loader, test_ids


def predict(model, test_loader, test_ids):
    """
    Generates predictions for the test set and saves to CSV.
    """
    device = get_device()
    model.eval()

    predictions = []
    print("Generating predictions on test set...")

    with torch.no_grad():
        for (X_batch,) in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())

    # Map predictions back to 1-7 range (model outputs 0-6)
    final_preds = np.array(predictions) + 1

    # Create Submission DataFrame
    submission = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: final_preds})

    # Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
