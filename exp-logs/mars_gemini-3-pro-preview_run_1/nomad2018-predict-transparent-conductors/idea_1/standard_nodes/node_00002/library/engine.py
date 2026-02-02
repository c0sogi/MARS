import torch
import torch.nn as nn
import numpy as np
from library.utils import compute_rmsle, save_submission


class Engine:
    """
    Engine class to manage the training, validation, and prediction processes.
    """

    def __init__(self, model, optimizer, device, scheduler=None):
        """
        Args:
            model: The PyTorch model to train.
            optimizer: The optimizer.
            device: The device (CPU or CUDA) to run on.
            scheduler: Learning rate scheduler (optional).
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = nn.MSELoss()
        self.best_val_rmsle = float("inf")
        self.best_model_state = None

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch)
            targets = batch.y

            # Ensure shapes match (batch.y might be [B, 1, 2] or [B, 2])
            if targets.dim() == 3:
                targets = targets.squeeze(1)

            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch.num_graphs

        return running_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns:
            val_loss: Mean Squared Error on log-transformed targets.
            val_rmsle: RMSLE on original scale targets.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds_log = []
        all_targets_log = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                targets = batch.y

                if targets.dim() == 3:
                    targets = targets.squeeze(1)

                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * batch.num_graphs

                all_preds_log.append(outputs.cpu())
                all_targets_log.append(targets.cpu())

        epoch_loss = running_loss / len(val_loader.dataset)

        val_rmsle = float("inf")
        if len(all_preds_log) > 0:
            # Concatenate all batches
            all_preds_log = torch.cat(all_preds_log, dim=0)
            all_targets_log = torch.cat(all_targets_log, dim=0)

            # Recover original scale for ground truth to compute RMSLE
            # The dataset provides log(1+x), so we use expm1 to invert
            all_targets_orig = torch.expm1(all_targets_log).numpy()

            # Compute RMSLE using the utility function
            val_rmsle = compute_rmsle(all_targets_orig, all_preds_log)

        # Scheduler step based on validation loss
        if self.scheduler:
            self.scheduler.step(epoch_loss)

        # Checkpointing based on RMSLE
        if val_rmsle < self.best_val_rmsle:
            self.best_val_rmsle = val_rmsle
            # Save state dict to CPU to save GPU memory
            self.best_model_state = {
                k: v.cpu() for k, v in self.model.state_dict().items()
            }

        return epoch_loss, val_rmsle

    def fit(self, train_loader, val_loader, num_epochs, early_stopping_patience):
        """
        Runs the full training loop with early stopping.
        """
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_rmsle = self.validate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val RMSLE: {val_rmsle}"
            )

            # Early stopping check (best_val_rmsle is updated in validate)
            # We check if the current val_rmsle matches the best to reset patience
            if val_rmsle <= self.best_val_rmsle:
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Best Validation RMSLE: {self.best_val_rmsle}")

        # Load best model weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model weights.")

    def predict(self, loader):
        """
        Generates predictions for a given loader.
        """
        self.model.eval()
        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)

                if hasattr(batch, "id"):
                    all_ids.extend(batch.id.cpu().numpy())

                all_preds.append(outputs.cpu())

        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds, dim=0)

        return all_ids, all_preds

    def generate_submission(self, test_loader, output_path):
        """
        Generates predictions for the test set and saves them to a CSV file.
        """
        print("Generating predictions for test set...")
        ids, preds_log = self.predict(test_loader)
        save_submission(ids, preds_log, output_path)
