import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.model import TriBranchWDCNet


class Trainer:
    """
    Manages the training, validation, and optimization of the Tri-Branch WDC Network.
    Implements aggressive scheduling and strict early stopping with state restoration.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimization Strategy
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function with Label Smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        # Aggressive Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",  # Monitor Loss
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.SCHEDULER_MIN_LR,
        )

        # Early Stopping State
        self.patience = Config.EARLY_STOPPING_PATIENCE
        self.best_val_acc = -np.inf
        self.best_model_state = None
        self.counter = 0
        self.early_stop = False

    def train_epoch(self):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target, _) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        """Runs validation phase."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target, _ in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, target)

                running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, epochs):
        """
        Main training loop with Early Stopping and Scheduler.
        """
        print(f"Starting training on device: {self.device}")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Update Scheduler based on Validation Loss
            old_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]["lr"]

            # Print Metrics
            print(
                f"Epoch {epoch}/{epochs} | "
                f"LR: {old_lr} | "
                f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
                f"Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # Early Stopping Logic (Monitor Accuracy)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.counter = 0
                # Deep copy the model state to ensure we save the exact best weights
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                # print(f"New best model found! Accuracy: {val_acc}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    self.early_stop = True
                    break

        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.2f} seconds.")
        print(f"Best Validation Accuracy: {self.best_val_acc}")

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)


def train_model(load_cached_data=True):
    """
    Orchestrates the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _, input_dim = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    print(f"Initializing TriBranchWDCNet with Input Dim: {input_dim}")
    # num_classes=7 because dataset targets are 0-6 (mapped from 1-7)
    model = TriBranchWDCNet(input_dim=input_dim, num_classes=7)
    model.to(device)

    # 4. Training
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit(epochs=Config.EPOCHS)

    # 5. Save Model
    print(f"Saving best model to {Config.MODEL_PATH}...")
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print("Model saved successfully.")

    return model
