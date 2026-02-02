import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from library.config import Config, set_seed
from library.dataset import SETIDataset
from library.model import ShallowCNN


class Trainer:
    """
    Manages the training, validation, and checkpointing of the Technosignature Detection model.
    """

    def __init__(self):
        """
        Initializes the Trainer with model, criterion, optimizer, and scheduler.
        """
        # Ensure reproducibility
        set_seed(Config.SEED)

        self.device = Config.DEVICE
        self.model = ShallowCNN().to(self.device)

        # Binary Cross Entropy with Logits (combines Sigmoid + BCELoss)
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Scheduler: Reduce LR when validation AUC stops improving
        # Mode 'max' because we want to maximize AUC
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.1, patience=1, verbose=False
        )

        self.best_auc = 0.0
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.

        Args:
            dataloader (DataLoader): The training data loader.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for images, targets in dataloader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Reshape targets to match model output (Batch, 1)
            targets = targets.view(-1, 1)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        return running_loss / count if count > 0 else 0.0

    def validate(self, dataloader):
        """
        Evaluates the model on the validation set.

        Args:
            dataloader (DataLoader): The validation data loader.

        Returns:
            tuple: (average_loss, auc_score)
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_targets = []
        all_probs = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Reshape targets
                targets = targets.view(-1, 1)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * images.size(0)
                count += images.size(0)

                # Apply sigmoid to logits to get probabilities for AUC
                probs = torch.sigmoid(logits)

                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        avg_loss = running_loss / count if count > 0 else 0.0

        # Concatenate all batches
        if len(all_targets) > 0:
            all_targets = np.concatenate(all_targets)
            all_probs = np.concatenate(all_probs)

            # Calculate AUC
            # Handle edge case where only one class is present in the batch
            if len(np.unique(all_targets)) > 1:
                auc = roc_auc_score(all_targets, all_probs)
            else:
                auc = 0.5
        else:
            auc = 0.5

        return avg_loss, auc

    def fit(self):
        """
        Main training loop handling datasets, epochs, early stopping, and saving.
        """
        print("Initializing Datasets...")
        train_dataset = SETIDataset(metadata_path=Config.TRAIN_METADATA)
        val_dataset = SETIDataset(metadata_path=Config.VAL_METADATA)

        # Debug mode: subset datasets
        if Config.DEBUG:
            print(
                f"Debug mode enabled. Training on first {Config.DEBUG_SAMPLE_SIZE} samples."
            )
            train_indices = list(
                range(min(len(train_dataset), Config.DEBUG_SAMPLE_SIZE))
            )
            val_indices = list(range(min(len(val_dataset), Config.DEBUG_SAMPLE_SIZE)))
            train_dataset = Subset(train_dataset, train_indices)
            val_dataset = Subset(val_dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print(
            f"Starting training for {Config.EPOCHS} epochs on device {self.device}..."
        )

        early_stopping_counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            # Training Step
            train_loss = self.train_epoch(train_loader)

            # Validation Step
            val_loss, val_auc = self.validate(val_loader)

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch}/{Config.EPOCHS} - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss} - "
                f"Val AUC: {val_auc}"
            )

            # Scheduler Step
            self.scheduler.step(val_auc)

            # Checkpointing and Early Stopping
            if val_auc > self.best_auc:
                print(
                    f"Validation AUC improved from {self.best_auc} to {val_auc}. Saving model..."
                )
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), self.best_model_path)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(
                    f"No improvement in Validation AUC. Early stopping counter: {early_stopping_counter}/{Config.PATIENCE}"
                )

            if early_stopping_counter >= Config.PATIENCE:
                print("Early stopping triggered. Training stopped.")
                break

        print(f"Training complete. Best Validation AUC: {self.best_auc}")
        print(f"Best model saved to: {self.best_model_path}")
