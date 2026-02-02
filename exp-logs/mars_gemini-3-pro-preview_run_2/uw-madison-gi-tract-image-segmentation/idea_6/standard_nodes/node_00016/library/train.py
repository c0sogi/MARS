import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, dice_coefficient
from library.dataset import process_metadata, UWDataset, get_transforms
from library.model import LinkNet, BCEDiceLoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    """
    Executes one validation epoch.
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            # Compute Dice Metric
            preds = torch.sigmoid(outputs)
            preds = (preds > Config.PRED_THRESHOLD).float()

            # dice_coefficient returns a scalar tensor
            d = dice_coefficient(preds, masks)
            running_dice += d.item() * batch_size

            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    epoch_dice = running_dice / dataset_size if dataset_size > 0 else 0.0

    return epoch_loss, epoch_dice


def train():
    """
    Main training loop with Early Stopping.
    """
    set_seed(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Data Preparation
    # process_metadata handles caching internally
    print("Loading and processing training metadata...")
    train_df = process_metadata(Config.TRAIN_METADATA_PATH, mode="train")

    print("Loading and processing validation metadata...")
    val_df = process_metadata(Config.VAL_METADATA_PATH, mode="val")

    # Datasets
    train_dataset = UWDataset(
        train_df, mode="train", transforms=get_transforms("train")
    )
    val_dataset = UWDataset(val_df, mode="val", transforms=get_transforms("val"))

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model, Optimizer, Scheduler, Loss
    device = Config.DEVICE
    model = LinkNet().to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = BCEDiceLoss(
        bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT
    )

    # 3. Training Loop
    best_dice = 0.0
    patience = 5
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step()

        # Print full precision metrics
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Dice: {val_dice}")

        # Early Stopping and Checkpointing
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with Dice: {best_dice}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation Dice: {best_dice}")
