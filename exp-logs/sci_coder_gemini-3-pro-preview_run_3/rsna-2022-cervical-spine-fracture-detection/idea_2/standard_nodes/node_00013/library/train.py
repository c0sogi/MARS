import os
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import ResNet18MIL
from library.loss import HierarchicalCompoundLoss
from library.utils import competition_log_loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        # inputs: (Batch, Seq, 3, H, W)
        # targets: (Batch, 8)

        inputs = inputs.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)

        batch_size = inputs.size(0)

        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and the competition metric.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            batch_size = inputs.size(0)

            # Forward pass
            logits = model(inputs)

            # Compute loss
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Collect predictions for metric calculation
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / dataset_size

    # Concatenate all batches
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        # Calculate competition metric
        val_metric = competition_log_loss(all_targets, all_preds)
    else:
        val_metric = float("inf")

    return val_loss, val_metric


def train_model(epochs=Config.EPOCHS, load_cached_data=True):
    """
    Main training loop with Early Stopping.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Setup directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Initialize Model
    print("Initializing model...")
    model = ResNet18MIL(
        backbone_name=Config.BACKBONE, pretrained=True, num_classes=Config.N_CLASSES
    )
    model = model.to(device)

    # Initialize Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR)

    # Initialize Loss
    criterion = HierarchicalCompoundLoss()

    # Get Dataloaders
    print("Loading data...")
    # The get_dataloaders function handles dataset creation.
    # Caching is handled internally by the Dataset class using Config.WORKING_DIR.
    train_loader, val_loader = get_dataloaders(
        train_metadata_path=Config.TRAIN_METADATA,
        val_metadata_path=Config.VAL_METADATA,
        image_dir=Config.TRAIN_IMAGES_DIR,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Training Loop
    best_metric = float("inf")
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print metrics with full precision
        print(
            f"Epoch {epoch}/{epochs} | LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Val Metric: {val_metric:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with metric: {best_metric:.10f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Metric: {best_metric:.10f}")
    return best_model_path
