import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from library.config import Config
from library.loss import BCEDiceLoss
from library.model import EfficientNetFPN
from library.data import get_dataloaders


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, masks in loader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        batch_size = images.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, loss_fn, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            batch_size = images.size(0)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            running_loss += loss.item() * batch_size

            # Calculate Dice Score
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            # Flatten for dice calculation
            preds = preds.view(batch_size, -1)
            targets = masks.view(batch_size, -1)

            intersection = (preds * targets).sum(dim=1)
            union = preds.sum(dim=1) + targets.sum(dim=1)

            # Dice coefficient per sample, avoiding division by zero
            dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
            running_dice += dice.sum().item()

            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    epoch_dice = running_dice / dataset_size

    return epoch_loss, epoch_dice


def train_model(epochs=Config.EPOCHS, load_cached_data=True, patience=5):
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    set_seed()
    device = torch.device(Config.DEVICE)

    # Initialize components
    model = EfficientNetFPN(
        encoder_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES
    )
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    loss_fn = BCEDiceLoss()

    # Get Dataloaders
    train_loader, val_loader = get_dataloaders(load_cached_data=load_cached_data)

    best_dice = -1.0
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice = validate(model, val_loader, loss_fn, device)

        # Step the scheduler
        scheduler.step()

        # Print metrics (full precision)
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Dice: {val_dice}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Checkpointing
        if val_dice > best_dice:
            print(
                f"Validation Dice improved from {best_dice} to {val_dice}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Dice: {best_dice}")
