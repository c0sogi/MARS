import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import random
from library import config, model, data


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device (CPU/GPU) to run on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        # Unpack batch
        # batched_atoms, batch_vec, batched_global, batched_targets, batched_ids
        atoms = batch[0].to(device)
        batch_indices = batch[1].to(device)
        glob_feats = batch[2].to(device)
        targets = batch[3].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(atoms, batch_indices, glob_feats)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss (MSE is averaged over batch, so multiply by batch size)
        running_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

    return running_loss / total_samples


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on a validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        float: Average validation loss (MSE on log-transformed targets).
        float: RMSLE metric (sqrt of MSE).
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            atoms = batch[0].to(device)
            batch_indices = batch[1].to(device)
            glob_feats = batch[2].to(device)
            targets = batch[3].to(device)

            outputs = model(atoms, batch_indices, glob_feats)

            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

    avg_loss = running_loss / total_samples
    rmsle = np.sqrt(avg_loss)
    return avg_loss, rmsle


class Trainer:
    """
    Manages the training process including early stopping and model saving.
    """

    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Configure optimizer and scheduler from the model definition
        self.optimizer, self.scheduler = self.model.configure_optimizers()

        # Loss function: MSE on log1p transformed targets
        self.criterion = nn.MSELoss()

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_path = config.MODEL_PATH

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs=config.NUM_EPOCHS,
        patience=config.PATIENCE,
    ):
        """
        Runs the training loop.
        """
        set_seed(config.SEED)
        print(f"Starting training on device: {self.device}")

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(
                self.model, train_loader, self.criterion, self.optimizer, self.device
            )

            val_loss, val_rmsle = evaluate(
                self.model, val_loader, self.criterion, self.device
            )

            # Step the scheduler based on validation loss
            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val RMSLE: {val_rmsle}"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved to {self.best_model_path}")
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience: {self.patience_counter}/{patience}")

            if self.patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training completed. Best Val Loss: {self.best_val_loss}")


def generate_submission(model, test_loader, device, output_path=config.SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the test set.
        device: Device to run inference on.
        output_path: Path to save the submission CSV.
    """
    # Load best model weights
    if os.path.exists(config.MODEL_PATH):
        print(f"Loading best model from {config.MODEL_PATH}")
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    else:
        print("Warning: No checkpoint found. Using current model weights.")

    model.eval()
    model.to(device)

    all_ids = []
    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            atoms = batch[0].to(device)
            batch_indices = batch[1].to(device)
            glob_feats = batch[2].to(device)
            # batch[3] contains placeholder targets, ignore
            ids = batch[4]

            outputs = model(atoms, batch_indices, glob_feats)

            # Move to CPU and numpy
            preds = outputs.cpu().numpy()

            # Inverse transform targets: expm1(y)
            # Since we trained on log1p(y), we need to reverse this for submission
            preds_original_scale = np.expm1(preds)

            all_ids.extend(ids.numpy())
            all_preds.append(preds_original_scale)

    # Concatenate all predictions
    all_preds = np.concatenate(all_preds, axis=0)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": all_ids,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    # Sort by ID to ensure correct order (though not strictly required by CSV format, it's good practice)
    submission_df.sort_values("id", inplace=True)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
