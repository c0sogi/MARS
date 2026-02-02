import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import from the provided library
from library.config import (
    WORKING_DIR,
    MODEL_SAVE_PATH,
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    POS_WEIGHT,
    EPOCHS,
    SEED,
)
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders
from library.model import SpatialSiameseModel


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Handles the training for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        # Unpack batch
        images = batch["image"].to(device, non_blocking=True)
        contralateral = batch["contralateral"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        batch_size = images.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images, contralateral)

        # Loss calculation
        # logits shape: [B, 1], labels shape: [B] -> unsqueeze labels to [B, 1]
        loss = criterion(logits, labels.unsqueeze(1))

        # Backward pass
        loss.backward()

        # Optimizer step (No gradient clipping as per strategy)
        optimizer.step()

        # Statistics
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Probabilistic F1 score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            contralateral = batch["contralateral"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            batch_size = images.size(0)

            # Forward pass
            logits = model(images, contralateral)
            loss = criterion(logits, labels.unsqueeze(1))

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Store for metrics
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    y_pred = np.concatenate(all_probs).flatten()
    y_true = np.concatenate(all_labels).flatten()

    # Calculate pF1
    pf1 = probabilistic_f1(y_true, y_pred)

    return epoch_loss, pf1


def run_training(load_cached_data=True, epochs=EPOCHS, patience=3):
    """
    Main training pipeline.

    Args:
        load_cached_data (bool): Whether to load pre-processed metadata from cache.
        epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience.
    """
    print(f"Starting training on device: {DEVICE}")

    # 1. Reproducibility
    seed_everything(SEED)

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Model Initialization
    print("Initializing model...")
    model = SpatialSiameseModel()
    model.to(DEVICE)

    # 4. Loss Function
    # Using aggressive positive weighting for 1:47 imbalance
    pos_weight_tensor = torch.tensor([POS_WEIGHT]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # 5. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 6. Training Loop
    best_pf1 = -1.0
    patience_counter = 0

    # Ensure working directory exists for saving model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{epochs} [LR: {current_lr}]")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, criterion, DEVICE)

        # Step Scheduler
        scheduler.step()

        # Print metrics (Full precision)
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val pF1: {val_pf1}")

        # Checkpointing & Early Stopping
        if val_pf1 > best_pf1:
            print(
                f"Validation pF1 improved from {best_pf1} to {val_pf1}. Saving model..."
            )
            best_pf1 = val_pf1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"\nTraining complete. Best Validation pF1: {best_pf1}")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")
