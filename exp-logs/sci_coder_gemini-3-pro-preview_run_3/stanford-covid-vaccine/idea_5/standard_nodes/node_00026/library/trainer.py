import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed, MCRMSELoss, calculate_metric
from library.model import DilatedResidualBiGRU


class Trainer:
    """
    Trainer class to manage the training, validation, and inference lifecycle
    of the RNA Degradation Prediction model.
    """

    def __init__(self, model=None):
        """
        Initialize the Trainer.

        Args:
            model: Optional pre-instantiated model. If None, instantiates DilatedResidualBiGRU.
        """
        # Ensure reproducibility
        set_seed(Config.SEED)

        self.device = Config.DEVICE

        # Initialize Model
        if model is None:
            self.model = DilatedResidualBiGRU()
        else:
            self.model = model

        self.model.to(self.device)

        # Initialize Optimizer
        # Using AdamW as specified in the strategy
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=0.01,  # Standard default for AdamW
        )

        # Initialize Loss Function
        self.criterion = MCRMSELoss()

        # Initialize Scheduler
        # Cosine Annealing over the total number of epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS
        )

        # Training State
        self.best_metric = float("inf")
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        num_samples = 0

        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Accumulate loss (weighted by batch size)
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

        avg_loss = running_loss / num_samples
        return avg_loss

    def validate(self, val_loader):
        """
        Runs validation on the provided loader.
        Returns the average loss (on all targets) and the MCRMSE metric (on scored targets).
        """
        self.model.eval()
        running_loss = 0.0
        num_samples = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.criterion(outputs, targets)

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                num_samples += batch_size

                # Store predictions and targets for metric calculation
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_loss = running_loss / num_samples

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate competition metric (MCRMSE on scored columns only)
        metric = calculate_metric(all_preds, all_targets)

        return avg_loss, metric

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        print(f"Starting training on {self.device} for {epochs} epochs...")

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_metric = self.validate(val_loader)

            # Update Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Print Metrics (Full Precision)
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Learning Rate: {current_lr}")
            print(f"  Train Loss: {train_loss}")
            print(f"  Val Loss:   {val_loss}")
            print(f"  Val MCRMSE: {val_metric}")

            # Checkpointing based on Val MCRMSE (Scored Columns)
            if val_metric < self.best_metric:
                print(
                    f"  [Improvement] Val MCRMSE decreased from {self.best_metric} to {val_metric}. Saving model..."
                )
                self.best_metric = val_metric
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
            else:
                self.patience_counter += 1
                print(
                    f"  [No Improvement] Patience: {self.patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            # Early Stopping
            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val MCRMSE: {self.best_metric}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.

        Returns:
            preds: Numpy array of shape (N_samples, Seq_Len, 5)
        """
        # Load best model weights
        if os.path.exists(Config.MODEL_PATH):
            print(f"Loading best model from {Config.MODEL_PATH}...")
            state_dict = torch.load(Config.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                "Warning: No saved model found at Config.MODEL_PATH. Using current model state."
            )

        self.model.eval()
        all_preds = []

        print("Starting inference...")
        with torch.no_grad():
            for inputs, ids in test_loader:
                inputs = inputs.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                all_preds.append(outputs.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        print(f"Inference complete. Output shape: {preds.shape}")
        return preds
