import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library.utils import set_seed, get_device
from library.dataset import load_dataset
from library.model import MGMTNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to logits to get probabilities
            probs = torch.sigmoid(outputs)

            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    # Calculate ROC AUC
    # Handle edge case where only one class is present in the batch/subset
    try:
        val_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training(
    num_epochs=30,
    batch_size=16,
    patience=10,
    load_cached_data=True,
    learning_rate=1e-4,
    save_dir="./working/idea_10",
):
    """
    Main function to run the training pipeline.

    Args:
        num_epochs (int): Maximum number of epochs.
        batch_size (int): Batch size for DataLoaders.
        patience (int): Early stopping patience.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        learning_rate (float): Learning rate for Adam optimizer.
        save_dir (str): Directory to save the best model.
    """
    # 1. Setup
    set_seed(42)
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")

    # 2. Data Loading
    print("Loading datasets...")
    train_dataset = load_dataset("train", load_cached_data=load_cached_data)
    val_dataset = load_dataset("val", load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 3. Model & Optimizer
    model = MGMTNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    best_val_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch}/{num_epochs} - Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Checkpoint & Early Stopping
        if val_auc > best_val_auc:
            print(
                f"Validation AUC improved ({best_val_auc} -> {val_auc}). Saving model..."
            )
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_val_auc}")
    return best_val_auc
