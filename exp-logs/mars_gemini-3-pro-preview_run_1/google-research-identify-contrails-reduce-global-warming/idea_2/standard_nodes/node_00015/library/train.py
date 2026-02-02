import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.dataset import ContrailDataset
from library.model import ResNet34UNet
from library.loss import CombinedLoss
from library.utils import set_seed


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Validates the model and calculates Global Dice score.
    """
    model.eval()
    running_loss = 0.0

    # Accumulators for Global Dice
    intersection_sum = 0.0
    union_sum = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # Calculate intersection and union for Global Dice
            # Apply sigmoid and threshold
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Flatten tensors to 1D for accumulation
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            intersection_sum += (preds_flat * masks_flat).sum().item()
            union_sum += preds_flat.sum().item() + masks_flat.sum().item()

    epoch_loss = running_loss / len(loader.dataset)

    # Global Dice Calculation
    # Formula: 2 * |X n Y| / (|X| + |Y|)
    epsilon = 1e-6
    if union_sum == 0:
        global_dice = 1.0
    else:
        global_dice = (2.0 * intersection_sum) / (union_sum + epsilon)

    return epoch_loss, global_dice


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    debug_subset_size=Config.DEBUG_SUBSET_SIZE,
    early_stopping_patience=10,
):
    """
    Main function to run the training pipeline.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        debug_subset_size (int, optional): Number of samples to use for debugging.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = ContrailDataset(split="train", debug_subset_size=debug_subset_size)
    val_dataset = ContrailDataset(
        split="validation", debug_subset_size=debug_subset_size
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

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 3. Model Initialization
    model = ResNet34UNet(
        in_channels=Config.IN_CHANNELS, out_channels=Config.CLASSES, pretrained=True
    )
    model.to(device)

    # 4. Optimization
    criterion = CombinedLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # 5. Training Loop
    best_dice = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]["lr"]

        # Print Metrics (Full Precision)
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Global Dice: {val_dice}")
        print(f"Learning Rate: {current_lr}")

        # Checkpoint and Early Stopping
        if val_dice > best_dice:
            print(
                f"Validation Dice improved from {best_dice} to {val_dice}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{early_stopping_patience}"
            )

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Global Dice: {best_dice}")
    print(f"Best model saved to: {best_model_path}")
