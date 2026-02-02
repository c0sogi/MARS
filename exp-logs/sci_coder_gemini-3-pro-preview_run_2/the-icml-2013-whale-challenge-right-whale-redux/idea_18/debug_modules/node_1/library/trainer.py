import os
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import calculate_metrics


class ModelTrainer:
    """
    Manages the training lifecycle for a single fold.
    Handles training loop, validation, optimization, and checkpointing
    based on the specified objective (AUC or Loss).
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        objective="auc",
        save_name="model",
    ):
        """
        Args:
            model: The PyTorch model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer: Optimizer instance.
            scheduler: Learning rate scheduler.
            device: 'cuda' or 'cpu'.
            objective (str): 'auc' or 'loss'. Determines the metric for checkpointing.
            save_name (str): Filename base for saving checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.objective = objective
        self.save_name = save_name

        # Binary Cross Entropy with Logits (more stable than BCELoss + Sigmoid)
        self.criterion = nn.BCEWithLogitsLoss()

        # Checkpoint directory
        self.checkpoint_dir = Config.CHECKPOINT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Tracking best metrics
        self.best_val_loss = float("inf")
        self.best_val_auc = float("-inf")

    def train_one_epoch(self):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            # Ensure targets are [Batch, 1] to match model output
            targets = targets.to(self.device).unsqueeze(1)

            batch_size = inputs.size(0)
            dataset_size += batch_size

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
        return epoch_loss

    def validate(self):
        """Evaluates the model on the validation set."""
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                batch_size = inputs.size(0)
                dataset_size += batch_size

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * batch_size

                # Apply sigmoid to get probabilities for AUC calculation
                probs = torch.sigmoid(outputs)

                all_targets.append(targets.cpu().numpy())
                all_preds.append(probs.cpu().numpy())

        val_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

        # Concatenate for metric calculation
        if len(all_targets) > 0:
            all_targets = np.concatenate(all_targets)
            all_preds = np.concatenate(all_preds)
            val_auc = calculate_metrics(all_targets, all_preds)
        else:
            val_auc = 0.5

        return val_loss, val_auc

    def fit(self, epochs=Config.EPOCHS, patience=Config.PATIENCE):
        """
        Runs the full training loop with Early Stopping.

        Args:
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.
        """
        print(
            f"Starting training for {self.save_name} | Objective: {self.objective} | Epochs: {epochs} | Patience: {patience}"
        )

        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_auc = self.validate()

            # Step the scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log metrics (Full precision)
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Checkpoint Logic based on Objective
            improved = False
            if self.objective == "auc":
                # Maximize AUC
                if val_auc > self.best_val_auc:
                    self.best_val_auc = val_auc
                    improved = True
            elif self.objective == "loss":
                # Minimize Loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    improved = True

            if improved:
                print(f"Metric improved. Saving checkpoint to {self.save_name}.pth")
                self.save_checkpoint()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Load the best model weights before finishing
        self.load_best_model()

    def save_checkpoint(self):
        """Saves the model state dict."""
        path = os.path.join(self.checkpoint_dir, f"{self.save_name}.pth")
        torch.save(self.model.state_dict(), path)

    def load_best_model(self):
        """Loads the best model state dict from disk."""
        path = os.path.join(self.checkpoint_dir, f"{self.save_name}.pth")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Restored best model from {path}")
        else:
            print(f"Warning: Checkpoint {path} not found.")

    def predict(self, loader):
        """
        Generates predictions for a given loader.
        Useful for generating OOF predictions or Test predictions.

        Returns:
            np.array: Predicted probabilities.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())

        if len(all_preds) > 0:
            return np.concatenate(all_preds)
        return np.array([])
