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
    Performs one epoch of training.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on (cuda/cpu).

    Returns:
        avg_loss: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return avg_loss


def validate_one_epoch(model, loader, criterion, device):
    """
    Performs validation on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        avg_loss: Average validation loss.
        auc_score: ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Concatenate all batches
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)

        # Calculate AUC
        # Handle edge case where only one class is present in validation batch
        try:
            auc_score = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auc_score = 0.5

        # Sanitize nan values (Cite debug_lesson_1)
        if np.isnan(auc_score):
            auc_score = 0.5
    else:
        auc_score = 0.5

    return avg_loss, auc_score


def run_training(load_cached=True):
    """
    Orchestrates the training process.

    Args:
        load_cached (bool): Whether to load pre-processed data from cache.

    Returns:
        best_auc (float): The best validation AUC achieved.
    """
    # 1. Setup
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = data_loader.get_dataloaders(load_cached=load_cached)

    # 3. Model Initialization
    print(f"Initializing model: {config.MODEL_NAME}")
    net = model.MontageEfficientNet(
        model_name=config.MODEL_NAME,
        pretrained=config.PRETRAINED,
        num_classes=config.NUM_CLASSES,
        drop_rate=config.DROPOUT_RATE,
    )
    net.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    checkpoint_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    start_time = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate_one_epoch(net, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{config.EPOCHS} | "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.15f}"
        )

        # Early Stopping & Checkpointing
        # Check if improvement is significant (greater than MIN_DELTA)
        if val_auc > (best_auc + config.MIN_DELTA):
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            state = {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
            }
            utils.save_checkpoint(state, checkpoint_path)
            print(f"--> New best model saved! (AUC: {best_auc:.15f})")
        else:
            patience_counter += 1
            print(f"--> No improvement. Patience: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s")
    print(f"Best Validation AUC: {best_auc:.15f} at Epoch {best_epoch}")

    return best_auc
