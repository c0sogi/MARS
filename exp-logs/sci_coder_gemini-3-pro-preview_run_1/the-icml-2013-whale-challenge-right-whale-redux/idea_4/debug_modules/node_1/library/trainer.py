import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
)
from library.utils import MetricMonitor, save_checkpoint, load_checkpoint
from library.dataset import mixup_data
from library.model import TimePreservingEfficientNet


class Trainer:
    """
    Trainer class to handle training, validation, and inference for the Whale Detection task.
    """

    def __init__(self, train_loader, val_loader, test_loader, test_ids):
        """
        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            test_loader (DataLoader): DataLoader for test data.
            test_ids (np.ndarray): Array of test clip IDs corresponding to test_loader.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.test_ids = test_ids

        self.device = DEVICE
        self.working_dir = WORKING_DIR
        self.best_model_path = os.path.join(self.working_dir, "best_model.pth")

        # Initialize Model
        self.model = TimePreservingEfficientNet()
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3
        )

        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

        self.best_score = -float("inf")

    def train_one_epoch(self, epoch):
        """
        Trains the model for one epoch using Mixup.
        """
        self.model.train()
        metric_monitor = MetricMonitor()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            # Apply Mixup
            data, target_a, target_b, lam = mixup_data(data, target, device=self.device)

            # Forward Pass
            self.optimizer.zero_grad()
            output = self.model(data)

            # Calculate Loss with Mixup
            # Targets need to be reshaped to (Batch, 1) to match logits
            target_a = target_a.view(-1, 1)
            target_b = target_b.view(-1, 1)

            loss = lam * self.criterion(output, target_a) + (1 - lam) * self.criterion(
                output, target_b
            )

            # Backward Pass
            loss.backward()
            self.optimizer.step()

            metric_monitor.update("Loss", loss.item())

        print(f"Epoch {epoch} | Train Loss: {metric_monitor.metrics['Loss']['avg']}")

    def validate(self, epoch):
        """
        Evaluates the model on the validation set.
        Returns:
            auc (float): The ROC-AUC score.
        """
        self.model.eval()
        metric_monitor = MetricMonitor()

        preds = []
        targets = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                # Forward Pass
                output = self.model(data)

                # Calculate Loss (Standard BCE)
                loss = self.criterion(output, target.view(-1, 1))
                metric_monitor.update("Loss", loss.item())

                # Calculate Probabilities for AUC
                probs = torch.sigmoid(output)

                preds.extend(probs.cpu().numpy().flatten())
                targets.extend(target.cpu().numpy().flatten())

        # Compute AUC
        try:
            auc = roc_auc_score(targets, preds)
        except ValueError:
            # Handle edge case if only one class is present in batch (unlikely with full val set)
            auc = 0.5

        print(
            f"Epoch {epoch} | Val Loss: {metric_monitor.metrics['Loss']['avg']} | Val AUC: {auc}"
        )
        return auc

    def fit(self, epochs=EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs on {self.device}...")

        patience = 5
        counter = 0

        for epoch in range(1, epochs + 1):
            self.train_one_epoch(epoch)
            val_score = self.validate(epoch)

            # Update Scheduler
            self.scheduler.step(val_score)

            # Checkpoint & Early Stopping
            if val_score > self.best_score:
                print(
                    f"Validation score improved ({self.best_score} -> {val_score}). Saving model..."
                )
                self.best_score = val_score
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_score, self.best_model_path
                )
                counter = 0
            else:
                counter += 1
                print(
                    f"No improvement. EarlyStopping counter: {counter} out of {patience}"
                )
                if counter >= patience:
                    print("Early stopping triggered.")
                    break

    def predict(self):
        """
        Generates predictions for the test set using the best saved model.
        Saves the result to SUBMISSION_PATH.
        """
        if not os.path.exists(self.best_model_path):
            print("No best model found. Skipping prediction.")
            return

        print("Loading best model for prediction...")
        load_checkpoint(self.model, None, self.best_model_path, device=self.device)
        self.model.eval()

        all_probs = []

        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.sigmoid(output)
                all_probs.extend(probs.cpu().numpy().flatten())

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"clip": self.test_ids, "probability": all_probs})

        # Save
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
