import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from library.config import Config
from library.utils import inverse_transform_targets


class Trainer:
    """
    Manages the training, validation, and checkpointing of the MSC-WDS model.
    """

    def __init__(self, model, optimizer, scheduler, device):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            device (torch.device): Computing device (CPU or GPU).
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_one_epoch(self, dataloader, epoch_index):
        """
        Executes one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        n_samples = 0

        for batch in dataloader:
            # Move batch data to device
            # Note: 'ids' are not tensors and are not needed for the forward pass
            inputs = {
                "atomic_features": batch["atomic_features"].to(self.device),
                "batch_index": batch["batch_index"].to(self.device),
                "global_features": batch["global_features"].to(self.device),
            }
            targets = batch["targets"].to(self.device)
            batch_size = targets.size(0)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Compute Loss (MSE on log-transformed targets)
            loss = nn.MSELoss()(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Accumulate loss (weighted by batch size)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        epoch_loss = running_loss / n_samples
        return epoch_loss

    def validate(self, dataloader):
        """
        Evaluates the model on the validation set.
        Returns the average loss and column-wise RMSLE metrics.
        """
        self.model.eval()
        running_loss = 0.0
        n_samples = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = {
                    "atomic_features": batch["atomic_features"].to(self.device),
                    "batch_index": batch["batch_index"].to(self.device),
                    "global_features": batch["global_features"].to(self.device),
                }
                targets = batch["targets"].to(self.device)
                batch_size = targets.size(0)

                # Forward pass
                outputs = self.model(inputs)

                # Compute Loss
                loss = nn.MSELoss()(outputs, targets)

                running_loss += loss.item() * batch_size
                n_samples += batch_size

                # Store predictions and targets for metric calculation
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        epoch_loss = running_loss / n_samples

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate RMSLE
        # Since targets are log(1+y) and model predicts log(1+y),
        # RMSLE is simply the RMSE of these values.
        squared_errors = (all_preds - all_targets) ** 2
        mse_per_col = np.mean(squared_errors, axis=0)
        rmsle_per_col = np.sqrt(mse_per_col)
        mean_rmsle = np.mean(rmsle_per_col)

        return epoch_loss, mean_rmsle, rmsle_per_col

    def fit(self, train_loader, val_loader, num_epochs, patience, checkpoint_path):
        """
        Runs the full training loop with early stopping.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs with patience {patience}...")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, val_rmsle, val_rmsle_cols = self.validate(val_loader)

            # Update Learning Rate
            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Logging (Full precision as requested)
            print(
                f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val RMSLE: {val_rmsle}"
            )
            print(f"  Val RMSLE per target: {val_rmsle_cols}")

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  New best model saved to {checkpoint_path}")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Load the best model weights before returning
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
            print("Training completed. Best model loaded.")
        else:
            print("Training completed. No checkpoint saved (did loss ever decrease?).")


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    Applies inverse transformation to recover original energy scale.
    """
    model.eval()
    ids = []
    preds = []

    print("Generating predictions for submission...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                "atomic_features": batch["atomic_features"].to(device),
                "batch_index": batch["batch_index"].to(device),
                "global_features": batch["global_features"].to(device),
            }
            batch_ids = batch["ids"]

            # Forward pass
            outputs = model(inputs)

            # Inverse transform: exp(y) - 1
            outputs_original = inverse_transform_targets(outputs.cpu().numpy())

            ids.extend(batch_ids)
            preds.append(outputs_original)

    # Combine all batches
    preds = np.concatenate(preds, axis=0)

    # Create DataFrame
    df = pd.DataFrame(preds, columns=Config.TARGET_COLS)
    df.insert(0, "id", ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
