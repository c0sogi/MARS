import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed
from library.data import prepare_datasets
from library.model import VolumetricEfficientNet


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to run on (cpu or cuda).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (Average validation loss, ROC AUC score)
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(x)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * x.size(0)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Handle edge case where only one class is present in the batch/dataset
    try:
        auc_score = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc_score = 0.5

    if np.isnan(auc_score):
        auc_score = 0.5

    return avg_loss, auc_score


def run_training(
    load_cached_data=True,
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
):
    """
    Main function to run the training pipeline.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        model: The best trained model.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")

    # Load Data
    train_dataset, val_dataset, _ = prepare_datasets(
        load_cached_data=load_cached_data, num_workers=Config.NUM_WORKERS
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = VolumetricEfficientNet()
    model.to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print("Starting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_auc}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")

    total_time = time.time() - start_time
    print(f"Training complete. Total time: {total_time:.2f}s")
    print(f"Best Validation AUC: {best_auc}")

    # Load best weights before returning
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model
