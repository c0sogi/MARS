import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, log_message, print_metric, get_device
from library.data_loader import get_dataloaders
from library.model import MGSHDNetwork


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for inputs, targets, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    avg_loss = running_loss / count if count > 0 else 0.0
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

            # Apply sigmoid to logits to get probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    # Concatenate all batches
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)

        # Calculate AUC
        # Handle edge case where only one class is present in the batch/set
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_probs)
        else:
            auc = 0.5
    else:
        auc = 0.5

    return avg_loss, auc


def run_training(load_cached_data=True, patience=5):
    """
    Main execution function to run the training pipeline.

    Args:
        load_cached_data (bool): Whether to load pre-processed .npy files from cache.
        patience (int): Number of epochs to wait for improvement before early stopping.
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = get_device()
    log_message(f"Using device: {device}")

    # 2. Data Loading
    log_message("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    if train_loader is None:
        log_message("Error: Training data not found. Aborting.")
        return

    # 3. Model Initialization
    log_message("Initializing Model...")
    model = MGSHDNetwork().to(device)

    # 4. Optimization
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    epochs_no_improve = 0

    log_message("Starting training loop...")

    for epoch in range(1, Config.EPOCHS + 1):
        log_message(f"Epoch {epoch}/{Config.EPOCHS}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print_metric("Training Loss", train_loss)

        # Validate
        if val_loader:
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            print_metric("Validation Loss", val_loss)
            print_metric("Validation AUC", val_auc)

            # Checkpointing
            if val_auc > best_auc:
                best_auc = val_auc
                epochs_no_improve = 0
                log_message(f"New best AUC! Saving model to {Config.MODEL_PATH}")
                torch.save(model.state_dict(), Config.MODEL_PATH)
            else:
                epochs_no_improve += 1
                log_message(
                    f"No improvement in AUC. Patience: {epochs_no_improve}/{patience}"
                )

            # Early Stopping
            if epochs_no_improve >= patience:
                log_message("Early stopping triggered.")
                break
        else:
            # If no validation set, just save the model every epoch or last epoch
            # Here we save every epoch as 'best' since we can't judge
            log_message(f"No validation set. Saving model to {Config.MODEL_PATH}")
            torch.save(model.state_dict(), Config.MODEL_PATH)

    log_message(f"Training complete. Best Validation AUC: {best_auc}")
