import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
from library.config import Config
from library.dataset import get_loaders
from library.model import get_model


class Trainer:
    """
    Trainer class to manage the training and validation of the Animal Classifier.
    """

    def __init__(self, debug=False, epochs=Config.EPOCHS):
        """
        Initialize the Trainer.

        Args:
            debug (bool): If True, uses a small subset of data for debugging.
            epochs (int): Number of training epochs.
        """
        # Ensure reproducibility
        Config.set_seed(Config.SEED)

        # Create necessary directories
        Config.make_dirs()

        self.device = torch.device(Config.DEVICE)
        self.epochs = epochs
        self.patience = Config.PATIENCE
        self.best_val_f1 = -1.0
        self.debug = debug

        # Initialize Model
        self.model = get_model(device=self.device, weights_path=None)

        # Initialize Data Loaders
        # Note: train_loader uses WeightedRandomSampler for class balancing
        self.train_loader, self.val_loader, _ = get_loaders(debug=self.debug)

        # Loss Function
        # We use standard CrossEntropyLoss. Class balancing is handled by the sampler.
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=Config.SCHEDULER_STEP_SIZE,
            gamma=Config.SCHEDULER_GAMMA,
        )

    def train_epoch(self):
        """
        Runs one epoch of training.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        epoch_loss = running_loss / count if count > 0 else 0.0
        return epoch_loss

    def validate(self):
        """
        Runs validation on the validation set.

        Returns:
            tuple: (Average validation loss, Macro F1 score)
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                count += images.size(0)

                # Get predictions
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / count if count > 0 else 0.0

        # Calculate Macro F1 Score
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        return val_loss, val_f1

    def fit(self):
        """
        Executes the training pipeline with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")
        print(f"Debug Mode: {self.debug}")
        print(f"Epochs: {self.epochs}")
        print(f"Patience: {self.patience}")

        patience_counter = 0

        for epoch in range(self.epochs):
            # Training Step
            train_loss = self.train_epoch()

            # Validation Step
            val_loss, val_f1 = self.validate()

            # Update Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Print Metrics (Full Precision)
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"Learning Rate: {current_lr}")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Macro F1: {val_f1}")

            # Checkpoint and Early Stopping Logic
            if val_f1 > self.best_val_f1:
                print(
                    f"Validation F1 improved from {self.best_val_f1} to {val_f1}. Saving model..."
                )
                self.best_val_f1 = val_f1
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"No improvement in Validation F1. Patience: {patience_counter}/{self.patience}"
                )

                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

            print("-" * 30)

        print(f"Training complete. Best Validation F1: {self.best_val_f1}")
        print(f"Best model saved to: {Config.MODEL_SAVE_PATH}")


def run_training(debug=False, epochs=Config.EPOCHS):
    """
    Helper function to instantiate the Trainer and start training.

    Args:
        debug (bool): Whether to run in debug mode (smaller dataset).
        epochs (int): Number of epochs to train.
    """
    trainer = Trainer(debug=debug, epochs=epochs)
    trainer.fit()
