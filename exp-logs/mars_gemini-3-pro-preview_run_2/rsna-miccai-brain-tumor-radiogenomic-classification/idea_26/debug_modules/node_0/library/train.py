import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed
from library.data import get_dataloader
from library.model import AsymmetricEfficientNet


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Ensure targets have the correct shape (B, 1) for BCEWithLogitsLoss
        targets = targets.unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass (outputs are logits)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions (sigmoid applied for AUC) and targets
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle case where only one class is present in the batch/epoch
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Ensure targets have the correct shape (B, 1)
            targets = targets.unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            preds = torch.sigmoid(outputs)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training(num_epochs=Config.NUM_EPOCHS, load_cached_data=True):
    """
    Main execution function for the training pipeline.

    Args:
        num_epochs (int): Number of training epochs.
        load_cached_data (bool): Whether to use cached ROI data.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure working directory exists for model saving
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Initializing AsymmetricEfficientNet on {device}...")
    model = AsymmetricEfficientNet()
    model = model.to(device)

    # 2. Data Loading
    print("Loading DataLoaders...")
    train_loader = get_dataloader("train", load_cached_data=load_cached_data)
    val_loader = get_dataloader("val", load_cached_data=load_cached_data)

    # 3. Optimization
    # Using AdamW with aggressive weight decay as per Idea 26 description
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # BCEWithLogitsLoss is numerically more stable than Sigmoid + BCELoss
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_val_auc = -1.0
    patience_counter = 0

    print("Starting training loop...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        duration = time.time() - start_time

        # Print metrics with full precision
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {duration:.2f}s")
        print(f"Train Loss: {train_loss} | Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss} | Val AUC: {val_auc}")

        # 5. Model Checkpointing & Early Stopping
        if val_auc > best_val_auc:
            print(
                f"Validation AUC improved from {best_val_auc} to {val_auc}. Saving model..."
            )
            best_val_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_val_auc}")
    print(f"Best model saved to: {Config.MODEL_SAVE_PATH}")
