import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from library.config import Config
from library.utils import set_seed
from library.dataset import UWMadisonDataset
from library.model import DeepLabV3Plus, dice_coef
from library.loss import BCEDiceLoss


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, loader, loss_fn, device):
    """
    Executes one validation epoch.
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    dataset_size = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            # Calculate Dice Score
            # Apply sigmoid and threshold
            preds = torch.sigmoid(outputs)
            preds = (preds > Config.CONFIDENCE_THRESHOLD).float()

            # dice_coef returns shape (B,), mean it
            dice = dice_coef(masks, preds).mean().item()

            running_loss += loss.item() * batch_size
            running_dice += dice * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    epoch_dice = running_dice / dataset_size
    return epoch_loss, epoch_dice


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    debug=False,
):
    """
    Main training pipeline.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for dataloaders.
        load_cached_data (bool): Whether to use cached metadata.
        debug (bool): If True, runs on a small subset of data.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 1. Prepare Datasets
    train_dataset = UWMadisonDataset(split="train", load_cached_data=load_cached_data)
    val_dataset = UWMadisonDataset(split="val", load_cached_data=load_cached_data)

    if debug:
        # Use a small subset for debugging
        indices = list(range(min(len(train_dataset), 50)))
        train_dataset = Subset(train_dataset, indices)
        val_indices = list(range(min(len(val_dataset), 20)))
        val_dataset = Subset(val_dataset, val_indices)
        print("Debug mode: using subset of data.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model Initialization
    model = DeepLabV3Plus(num_classes=Config.NUM_CLASSES).to(device)

    # 3. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    loss_fn = BCEDiceLoss()

    # 4. Training Loop
    best_dice = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice = validate_one_epoch(model, val_loader, loss_fn, device)

        # Step the scheduler
        scheduler.step()

        # Print metrics with full precision
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Dice: {val_dice}")

        # Checkpointing
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            print(f"New best model saved with Dice: {best_dice}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val Dice: {best_dice}")
