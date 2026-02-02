import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import set_seed, AverageMeter, GlobalDiceMeter, save_checkpoint
from library.losses import CompositeLoss
from library.dataset import ContrailDataset, get_transforms
from library.model import UNetPlusPlus


def train_one_epoch(loader, model, criterion, optimizer, scaler, device, epoch):
    """
    Handles the training of one epoch.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast():
            logits = model(images)
            loss = criterion(logits, targets)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    dice_meter = GlobalDiceMeter()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            # We can use autocast in validation too for speed
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets)

            # Update Loss
            losses.update(loss.item(), images.size(0))

            # Update Global Dice
            # Pass logits (model output) and targets
            # The meter handles thresholding/sigmoid internally as per implementation
            dice_meter.update(torch.sigmoid(logits), targets)

    score = dice_meter.get_score()
    return losses.avg, score


def run_training(max_epochs=Config.EPOCHS, patience=10):
    """
    Main function to run the training pipeline.

    Args:
        max_epochs (int): Maximum number of epochs to train.
        patience (int): Early stopping patience.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    print(f"Starting training on device: {device}")

    # 2. Data Loading
    train_dataset = ContrailDataset(
        metadata_csv_path=Config.TRAIN_METADATA_PATH,
        stage="train",
        transform=get_transforms("train"),
    )

    val_dataset = ContrailDataset(
        metadata_csv_path=Config.VAL_METADATA_PATH,
        stage="validation",
        transform=get_transforms("validation"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Model
    model = UNetPlusPlus(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.IN_CHANNELS,
        classes=Config.CLASSES,
        output_stride=Config.ENCODER_OUTPUT_STRIDE,
    )
    model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # Monitoring Dice score (higher is better)
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
        verbose=True,
    )

    criterion = CompositeLoss(
        weight_focal=Config.WEIGHT_FOCAL,
        weight_dice=Config.WEIGHT_DICE,
        focal_alpha=Config.FOCAL_ALPHA,
        focal_gamma=Config.FOCAL_GAMMA,
    )

    scaler = GradScaler()

    # 5. Training Loop
    best_dice = 0.0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, scaler, device, epoch
        )

        # Validate
        val_loss, val_dice = validate(val_loader, model, criterion, device)

        # Scheduler Step
        scheduler.step(val_dice)

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(f"Epoch {epoch}/{max_epochs} - Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Dice: {val_dice}")

        # Checkpointing
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            epochs_no_improve = 0
            print(f"New best Dice: {best_dice}")
        else:
            epochs_no_improve += 1

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_dice": best_dice,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            Config.CHECKPOINT_DIR,
        )

        # Early Stopping
        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {epochs_no_improve} epochs without improvement."
            )
            break

    print(f"Training complete. Best Validation Dice: {best_dice}")
