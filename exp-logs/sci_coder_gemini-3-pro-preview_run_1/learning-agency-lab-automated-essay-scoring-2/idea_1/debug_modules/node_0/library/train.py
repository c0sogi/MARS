import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.data import get_dataloaders
from library.model import DANRegressor


class Trainer:
    """
    Trainer class for the Deep Averaging Network (DAN) Regressor.
    Handles training loops, validation, early stopping, and model saving.
    """

    def __init__(self, config=Config):
        """
        Initialize the Trainer.

        Args:
            config: Configuration class containing hyperparameters and paths.
        """
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Initialize Model
        self.model = DANRegressor(config).to(self.device)

        # Initialize Optimizer
        # Using Adam as it works well for simple embeddings + MLP architectures
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # Initialize Loss Function
        # MSE is standard for regression to approximate the ordinal score
        self.criterion = nn.MSELoss()

        # Early Stopping tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.

        Args:
            train_loader: DataLoader for training data.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch in train_loader:
            # Move data to device
            input_ids = batch["input_ids"].to(self.device)
            scores = batch["scores"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # Output shape: (batch_size, 1) -> squeeze to (batch_size)
            outputs = self.model(input_ids).squeeze(-1)

            # Compute loss
            loss = self.criterion(outputs, scores)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Accumulate metrics
            running_loss += loss.item() * input_ids.size(0)
            count += input_ids.size(0)

        return running_loss / count if count > 0 else 0.0

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            dict: Dictionary containing 'loss' (MSE) and 'qwk' (Quadratic Weighted Kappa).
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                scores = batch["scores"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids).squeeze(-1)

                # Compute loss
                loss = self.criterion(outputs, scores)

                running_loss += loss.item() * input_ids.size(0)
                count += input_ids.size(0)

                # Store predictions and labels for QWK calculation
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(scores.cpu().numpy())

        avg_loss = running_loss / count if count > 0 else 0.0

        # Process predictions for QWK
        # 1. Clip to valid range [1, 6]
        # 2. Round to nearest integer
        preds_np = np.array(all_preds)
        labels_np = np.array(all_labels)

        preds_clipped = np.clip(preds_np, 1, 6)
        preds_rounded = np.round(preds_clipped).astype(int)
        labels_int = labels_np.astype(int)

        qwk_score = compute_qwk(labels_int, preds_rounded)

        return {"loss": avg_loss, "qwk": qwk_score}

    def save_model(self, path):
        """
        Saves the model state dictionary.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def fit(self, train_loader, val_loader):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics["loss"]
            val_qwk = val_metrics["qwk"]

            # Print Metrics (Full precision as requested)
            print(
                f"Epoch {epoch}/{self.config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss} - "
                f"Val QWK: {val_qwk}"
            )

            # Early Stopping Check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(self.config.MODEL_SAVE_PATH)
                print(
                    f"Validation loss improved. Model saved to {self.config.MODEL_SAVE_PATH}"
                )
            else:
                self.patience_counter += 1
                print(
                    f"No improvement in validation loss. Patience: {self.patience_counter}/{self.config.PATIENCE}"
                )

            if self.patience_counter >= self.config.PATIENCE:
                print("Early stopping triggered.")
                break


def run_training(load_cached_data=True):
    """
    Orchestrates the training process.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # 1. Set Reproducibility
    seed_everything(Config.SEED)

    # 2. Prepare Data
    # get_dataloaders handles caching internally based on the flag
    train_loader, val_loader, _, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Initialize Trainer
    trainer = Trainer(Config)

    # 4. Execute Training
    trainer.fit(train_loader, val_loader)

    print("Training pipeline completed.")
