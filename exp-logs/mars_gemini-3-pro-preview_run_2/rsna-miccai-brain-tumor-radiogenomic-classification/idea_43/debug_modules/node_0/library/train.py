import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed
from library.data_loader import get_dataloader
from library.model import AsymmetricEfficientNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to run on (cpu or cuda).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Ensure targets have shape (B, 1) to match model output
        targets = targets.unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (Average validation loss, ROC AUC score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Ensure targets have shape (B, 1)
            targets = targets.unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to logits to get probabilities for AUC
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Concatenate all batches
    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)

        # Calculate ROC AUC
        # Handle edge case where only one class is present
        if len(np.unique(all_targets)) > 1:
            auc_score = roc_auc_score(all_targets, all_probs)
        else:
            auc_score = 0.5
    else:
        auc_score = 0.5

    return epoch_loss, auc_score


def run_training(debug=False):
    """
    Main execution function for the training pipeline.
    Handles setup, training loop, logging, and early stopping.

    Args:
        debug (bool): If True, runs on a small subset of data.
    """
    # 1. Reproducibility
    set_seed(Config.SEED)

    # 2. Setup Device
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 3. Data Loading
    # The get_dataloader function handles caching internally
    print("Initializing DataLoaders...")
    train_loader = get_dataloader(
        split="train", batch_size=Config.BATCH_SIZE, debug=debug
    )
    val_loader = get_dataloader(split="val", batch_size=Config.BATCH_SIZE, debug=debug)

    # 4. Model Initialization
    print("Initializing Model...")
    model = AsymmetricEfficientNet(pretrained=True)
    model = model.to(device)

    # 5. Optimizer & Loss
    # AdamW with aggressive weight decay as per Idea 43
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        end_time = time.time()
        duration = end_time - start_time

        # Logging (Full precision as requested)
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} (Time: {duration:.2f}s)")
        print(f"    Train Loss: {train_loss}")
        print(f"    Val Loss:   {val_loss}")
        print(f"    Val AUC:    {val_auc}")

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            print(
                f"    Validation AUC improved ({best_auc} -> {val_auc}). Saving model to {Config.MODEL_PATH}..."
            )
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"    No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("    Early stopping triggered.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")
