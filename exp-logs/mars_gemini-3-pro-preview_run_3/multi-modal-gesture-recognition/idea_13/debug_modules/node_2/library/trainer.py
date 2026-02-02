import os
import time
import torch
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import get_dataloaders
from library.model import SK_ARN, CascadedLoss


class Trainer:
    """
    Manages the training lifecycle for the SK-ARN model, including training loops,
    validation, early stopping, and model checkpointing.
    """

    def __init__(self, debug_max=None):
        """
        Initialize the Trainer.

        Args:
            debug_max (int, optional): Maximum number of samples to load for debugging purposes.
                                       If None, loads the full dataset.
        """
        # Ensure reproducibility
        set_seed(Config.SEED)

        # Setup device
        self.device = get_device()

        # Load Data
        # We only need train and val loaders for the trainer
        self.train_loader, self.val_loader, _ = get_dataloaders(debug_max=debug_max)

        # Initialize Model
        self.model = SK_ARN().to(self.device)

        # Initialize Loss Function (CascadedLoss)
        self.criterion = CascadedLoss().to(self.device)

        # Initialize Optimizer (Adam)
        # Using standard Adam as per design (avoiding AdamW for Recurrent stability)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Training Configuration
        self.epochs = Config.NUM_EPOCHS
        self.patience = 10  # Early stopping patience
        self.best_val_loss = float("inf")
        self.counter = 0  # Counter for early stopping

    def train_epoch(self):
        """
        Executes one epoch of training.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            # Move data to device
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # outputs is a dict containing logits for stage1, stage2, stage3
            outputs = self.model(features)

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def fit(self):
        """
        Runs the full training process with Early Stopping.
        Saves the best model to Config.BEST_MODEL_PATH.
        """
        print(f"Starting training on device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 30)

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            # Run Training and Validation
            train_loss = self.train_epoch()
            val_loss = self.validate()

            duration = time.time() - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Time: {duration:.4f}s"
            )

            # Checkpoint & Early Stopping Logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                # Save best model
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(
                    f"Validation loss improved. Saved model to {Config.BEST_MODEL_PATH}"
                )
            else:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        print("-" * 30)
        print(f"Training finished. Best Validation Loss: {self.best_val_loss}")
