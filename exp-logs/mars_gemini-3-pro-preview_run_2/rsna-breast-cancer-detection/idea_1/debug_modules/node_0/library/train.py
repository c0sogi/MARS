import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from library.config import (
    DEVICE,
    LEARNING_RATE,
    EPOCHS,
    POS_WEIGHT,
    CACHE_DIR,
    seed_everything,
)
from library.data import get_dataloaders
from library.model import HybridEfficientNet
from library.utils import probabilistic_f1


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to run training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for (images, tabular), targets in loader:
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device).view(-1, 1)

        batch_size = images.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model((images, tabular))
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_epoch(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run evaluation on.

    Returns:
        tuple: (average_loss, pF1_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for (images, tabular), targets in loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device).view(-1, 1)

            batch_size = images.size(0)

            # Forward pass
            logits = model((images, tabular))
            loss = criterion(logits, targets)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store for global metric calculation
            all_targets.append(targets.cpu())
            all_preds.append(probs.cpu())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_targets = torch.cat(all_targets, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    # Calculate Probabilistic F1
    pf1 = probabilistic_f1(all_targets, all_preds)

    return epoch_loss, pf1.item()


def run_training(
    debug=False,
    load_cached_data=True,
    epochs=EPOCHS,
    batch_size_override=None,
    save_path=None,
):
    """
    Main function to run the training pipeline.

    Args:
        debug (bool): If True, runs on a subset of data.
        load_cached_data (bool): Whether to load pre-processed metadata from cache.
        epochs (int): Number of training epochs.
        batch_size_override (int, optional): Override default batch size if needed.
        save_path (str, optional): Path to save the best model. Defaults to CACHE_DIR/best_model.pth.
    """
    seed_everything()

    if save_path is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        save_path = os.path.join(CACHE_DIR, "best_model.pth")

    print(f"Starting training on device: {DEVICE}")
    print(f"Debug Mode: {debug}")
    print(f"Epochs: {epochs}")

    # 1. Get DataLoaders
    train_loader, val_loader, test_loader, num_tabular_features = get_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    # 2. Initialize Model
    model = HybridEfficientNet(
        num_tabular_features=num_tabular_features,
        backbone_name="efficientnet_b0",
        pretrained=True,
    )
    model = model.to(DEVICE)

    # 3. Setup Training Components
    # Weighted BCE Loss to handle class imbalance
    pos_weight_tensor = torch.tensor([POS_WEIGHT]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # 4. Training Loop
    best_pf1 = -1.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, DEVICE
        )

        # Validate
        val_loss, val_pf1 = validate_epoch(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val pF1: {val_pf1}")

        # Save Best Model
        if val_pf1 > best_pf1:
            print(
                f"Validation pF1 improved from {best_pf1} to {val_pf1}. Saving model..."
            )
            best_pf1 = val_pf1
            torch.save(model.state_dict(), save_path)
        else:
            print(f"Validation pF1 did not improve (Best: {best_pf1}).")

    print(f"\nTraining complete. Best Validation pF1: {best_pf1}")
    print(f"Best model saved to: {save_path}")

    return best_pf1
