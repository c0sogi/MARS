import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_device


class Trainer:
    """
    Encapsulates the training, validation, and inference logic for the Deep Pre-Activation Parallel DCN-ResNet.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        learning_rate: float = Config.LEARNING_RATE,
        weight_decay: float = Config.WEIGHT_DECAY,
    ):
        """
        Args:
            model: The PyTorch model to train.
            device: The device to run training on. Defaults to Config.DEVICE.
            learning_rate: Learning rate for the optimizer.
            weight_decay: Weight decay for the optimizer.
        """
        self.model = model
        self.device = device if device else get_device()
        self.model.to(self.device)

        # Criterion
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler: ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=Config.SCHEDULER_MODE,
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=True,
        )

        # State for best model
        self.best_model_state = None
        self.best_val_acc = 0.0

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
    ):
        """
        Orchestrates the full training process with Early Stopping and Scheduling.
        """
        print(f"Starting training for {epochs} epochs...")
        patience_counter = 0

        # Ensure directories exist
        os.makedirs(os.path.dirname(Config.MODEL_CHECKPOINT_PATH), exist_ok=True)

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
                f"Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # Scheduler Step
            self.scheduler.step(val_acc)

            # Early Stopping Logic
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                # Save checkpoint
                torch.save(self.best_model_state, Config.MODEL_CHECKPOINT_PATH)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. "
                        f"Best Val Acc: {self.best_val_acc}"
                    )
                    break

        # Load best weights into the model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Restored best model weights.")

    def predict(self, test_loader, test_ids, output_path=Config.SUBMISSION_PATH):
        """
        Generates predictions for the test set and saves them to a CSV file.
        Uses the current state of the model (assumed to be best weights after fit).
        """
        self.model.eval()
        predictions = []

        print("Generating predictions...")
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                # Convert back to 1-based indexing (0-6 -> 1-7)
                predicted = predicted + 1
                predictions.extend(predicted.cpu().numpy())

        if len(test_ids) != len(predictions):
            print(
                f"Warning: Number of IDs ({len(test_ids)}) does not match "
                f"predictions ({len(predictions)})"
            )

        # Create submission DataFrame
        submission_df = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
