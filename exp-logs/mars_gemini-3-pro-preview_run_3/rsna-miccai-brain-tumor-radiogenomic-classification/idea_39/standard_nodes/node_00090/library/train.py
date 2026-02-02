import torch
import torch.nn as nn
import numpy as np
import os
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.data_loader import get_dataloaders
from library.model import SSFNet


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model: The SSFNet model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for (even, odd), labels in dataloader:
        batch_size = labels.size(0)

        # Move data to device
        even = even.to(device)
        odd = odd.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(even, odd)

        # Compute loss (ensure logits are flattened to match labels)
        loss = criterion(logits.view(-1), labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The SSFNet model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        tuple: (Average Validation Loss, Validation AUC Score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for (even, odd), labels in dataloader:
            batch_size = labels.size(0)

            even = even.to(device)
            odd = odd.to(device)
            labels = labels.to(device)

            logits = model(even, odd)
            loss = criterion(logits.view(-1), labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Calculate AUC
    if len(np.unique(all_labels)) > 1:
        auc_score = calculate_roc_auc(np.array(all_labels), np.array(all_preds))
    else:
        # Handle edge case where validation set might have only one class
        auc_score = 0.5

    return avg_loss, auc_score


def run_training(
    num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True
):
    """
    Main function to run the training pipeline.

    Args:
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loaders.
        load_cached_data (bool): Whether to use cached numpy arrays for data.

    Returns:
        float: The best validation AUC achieved.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Setup device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Load Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # Initialize Model, Criterion, Optimizer
    print("Initializing Model...")
    model = SSFNet().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Training Loop Variables
    best_auc = 0.0
    patience_counter = 0
    best_model_path = Config.MODEL_PATH

    print("Starting Training...")
    print("-" * 50)

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics (Full precision)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss:   {val_loss}")
        print(f"Val AUC:    {val_auc}")

        # Checkpointing and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

        print("-" * 50)

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_auc
