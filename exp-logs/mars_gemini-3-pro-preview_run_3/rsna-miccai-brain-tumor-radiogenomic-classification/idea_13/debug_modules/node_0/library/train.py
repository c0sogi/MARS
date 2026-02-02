import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library import config
from library import utils
from library import data_loader
from library import model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)

        # Calculate loss
        loss = criterion(logits, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(inputs)

            # Calculate loss
            loss = criterion(logits, targets)
            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Store for metric calculation
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    # Calculate AUC
    # Handle edge case where only one class is present in the batch/subset
    try:
        auc_score = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc_score = 0.5

    return total_loss, auc_score


def run_training(max_samples=None):
    """
    Main execution function to train the model.

    Args:
        max_samples (int, optional): Limit dataset size for debugging.
    """
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = utils.get_device()
    print(f"Device: {device}")

    # 2. Data Loading
    # Pass max_samples to get_dataloaders for flexibility
    train_loader, val_loader, _, _ = data_loader.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=True,
        max_samples=max_samples,
    )

    # 3. Model Initialization
    net = model.MGMTNet()
    net = net.to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(net, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print metrics (Full precision as requested)
        print(f"Epoch {epoch+1}/{config.EPOCHS} - Time: {elapsed}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            print(
                f"Validation AUC improved from {best_auc} to {val_auc}. Saving model..."
            )
            best_auc = val_auc
            patience_counter = 0
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
