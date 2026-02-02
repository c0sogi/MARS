import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import (
    tqdm,
)  # Optional, but usually good for local tracking, though instruction says no progress bars. I will omit tqdm to be safe.

from library.config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SCHEDULER_MODE,
    EARLY_STOPPING_PATIENCE,
    SUBMISSION_PATH,
    ID_COL,
    TARGET_COL,
    NUM_CLASSES,
)
from library.utils import seed_everything, EarlyStopping
from library.data_loader import get_dataloaders
from library.model import WideAsymmetricDCNResNet


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer: AdamW (Decoupled Weight Decay)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Loss Function: Cross Entropy
        self.criterion = nn.CrossEntropyLoss()

        # Scheduler: ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=SCHEDULER_MODE,
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            verbose=False,  # Keep output clean
        )

        # Early Stopping
        self.early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE, mode=SCHEDULER_MODE
        )

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

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

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, epochs):
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            # Update Scheduler based on validation accuracy (mode='max')
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss} - Train Acc: {train_acc} - "
                f"Val Loss: {val_loss} - Val Acc: {val_acc} - "
                f"LR: {current_lr}"
            )

            # Check Early Stopping
            self.early_stopping(val_acc, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # Restore best weights
        print("Restoring best model weights...")
        self.early_stopping.restore_best_weights(self.model)

    def predict(self, test_loader):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)


def run_training():
    # 1. Setup
    seed_everything()

    # 2. Data Loading
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    print(f"Initializing Model (Input Dim: {input_dim}, Classes: {NUM_CLASSES})...")
    model = WideAsymmetricDCNResNet(input_dim=input_dim, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # 4. Training
    trainer = Trainer(model, train_loader, val_loader, DEVICE)
    trainer.fit(EPOCHS)

    # 5. Inference
    print("Generating predictions on test set...")
    preds = trainer.predict(test_loader)

    # 6. Post-processing
    # The model predicts 0-6, but target is 1-7. We must add 1.
    final_preds = preds + 1

    # 7. Submission
    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission_df = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_preds})

    # Ensure ID is int if it was loaded as float/int
    submission_df[ID_COL] = submission_df[ID_COL].astype(int)

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")

    # Validation check (head)
    print("Submission head:")
    print(submission_df.head())
