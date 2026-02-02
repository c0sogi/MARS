import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from library.config import Config
from library.model import BiGRUModel


class Trainer:
    """
    Trainer class to manage the training and evaluation of the BiGRUModel.
    """

    def __init__(self, model=None):
        self.device = torch.device(Config.DEVICE)

        # Initialize model
        if model is None:
            self.model = BiGRUModel().to(self.device)
        else:
            self.model = model.to(self.device)

        # Loss function: Mean Absolute Error (L1 Loss) for robustness
        self.criterion = nn.L1Loss()

        # Optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=Config.LEARNING_RATE
        )

        # Scheduler: ReduceLROnPlateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        # Early stopping state
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set.
        Returns average loss and MAE.
        """
        self.model.eval()
        running_loss = 0.0
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

                preds_list.append(outputs.cpu().numpy())
                targets_list.append(targets.cpu().numpy())

        epoch_loss = running_loss / len(dataloader.dataset)

        # Calculate MAE for detailed metric tracking
        all_preds = np.concatenate(preds_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)
        mae = mean_absolute_error(all_targets, all_preds)

        return epoch_loss, mae

    def fit(self, train_loader, val_loader):
        """
        Main training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae = self.evaluate(val_loader)

            # Step the scheduler based on validation loss
            self.scheduler.step(val_loss)

            duration = time.time() - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val MAE: {val_mae} | "
                f"Time: {duration}s"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print(f"  -> Model saved! Best Val Loss: {self.best_val_loss}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.PATIENCE:
                    print("  -> Early stopping triggered.")
                    break

        # Load best model state before returning
        if os.path.exists(Config.MODEL_PATH):
            self.model.load_state_dict(torch.load(Config.MODEL_PATH))

        return self.model
