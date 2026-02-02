import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from library.config import Config
from library.utils import do_kaggle_metric, unpad_image
from library.loss import BCELovaszLoss
from library.dataset import get_dataloaders
from library.model import SaltNet


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across numpy and torch.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model: The neural network.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, masks, depths, ids in loader:
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass: Model expects (images, depths)
        logits = model(images, depths)
        loss = criterion(logits, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    Calculates Loss and MAP.

    Args:
        model: The neural network.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        tuple: (average_loss, map_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks, depths, ids in loader:
            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)
            batch_size = images.size(0)

            logits = model(images, depths)
            loss = criterion(logits, masks)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(logits).cpu().numpy()
            true_masks = masks.cpu().numpy()

            # Unpad predictions and masks to original size (101x101) for accurate metric calculation
            for i in range(batch_size):
                # preds[i, 0] is (128, 128)
                p = unpad_image(preds[i, 0], Config.ORIG_SIZE)
                m = unpad_image(true_masks[i, 0], Config.ORIG_SIZE)

                all_preds.append(p)
                all_masks.append(m)

    epoch_loss = running_loss / dataset_size

    # Calculate MAP using the competition metric function
    # Inputs must be numpy arrays of shape (N, H, W)
    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)

    val_map = do_kaggle_metric(all_preds, all_masks, threshold=0.5)

    return epoch_loss, val_map


def train_model(config=Config):
    """
    Main training loop.
    Initializes model, optimizer, scheduler, and runs the training process.
    Handles checkpointing and early stopping.
    """
    set_seed(config.SEED)

    device = torch.device(config.DEVICE)
    print(f"Training on device: {device}")

    # Ensure output directories exist
    config.create_dirs()

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(config)

    # Initialize Model
    model = SaltNet()
    model = model.to(device)

    # Initialize Loss
    criterion = BCELovaszLoss(
        bce_weight=config.BCE_WEIGHT, lovasz_weight=config.LOVASZ_WEIGHT
    )

    # Initialize Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Initialize Scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN
    )

    # Training Loop Variables
    best_map = 0.0
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train and Validate
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_map = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Time: {elapsed:.2f}s | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MAP: {val_map}"
        )

        # Checkpointing
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)
            print(f"New Best Model Saved! MAP: {best_map}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training Complete. Best MAP: {best_map}")
