import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import AverageMeter, save_checkpoint
from library.data_loader import get_loader
from library.model import MSAHN


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Executes one training epoch.
    """
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()

    for i, (images, inc_angles, labels) in enumerate(train_loader):
        images = images.to(device)
        inc_angles = inc_angles.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images, inc_angles)
        loss = criterion(outputs, labels)

        # Compute accuracy
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == labels).float().mean()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))
        accs.update(acc.item(), images.size(0))

    return losses.avg, accs.avg


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        for images, inc_angles, labels in val_loader:
            images = images.to(device)
            inc_angles = inc_angles.to(device)
            labels = labels.to(device)

            outputs = model(images, inc_angles)
            loss = criterion(outputs, labels)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc = (preds == labels).float().mean()

            losses.update(loss.item(), images.size(0))
            accs.update(acc.item(), images.size(0))

    return losses.avg, accs.avg


def run_fold(fold_idx, train_df, val_df):
    """
    Orchestrates the training process for a single fold.
    Handles initialization, training loop, dynamic regularization, and early stopping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fold {fold_idx}: Starting training on device {device}")

    # Initialize Model
    model = MSAHN().to(device)

    # Initialize Criterion
    criterion = nn.BCEWithLogitsLoss()

    # Initialize Optimizer
    # Start with 0 weight decay; it will be adjusted dynamically based on loss comparison
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.0
    )

    # Initialize Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # Prepare Data Loaders
    train_loader = get_loader(
        train_df, batch_size=Config.BATCH_SIZE, shuffle=True, augment=True
    )
    val_loader = get_loader(
        val_df, batch_size=Config.BATCH_SIZE, shuffle=False, augment=False
    )

    # Early Stopping Variables
    best_loss = float("inf")
    best_acc = 0.0
    best_weights = None
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train Step
        train_loss, train_acc = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validation Step
        val_loss, val_acc = validate(val_loader, model, criterion, device)

        # Scheduler Step
        scheduler.step(val_loss)

        # Dynamic Weight Decay Logic
        # Apply weight decay only if Validation Loss > Training Loss
        current_wd = 0.0
        if val_loss > train_loss:
            current_wd = Config.WEIGHT_DECAY

        for param_group in optimizer.param_groups:
            param_group["weight_decay"] = current_wd

        # Logging
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed}s")
        print(f"Train Loss: {train_loss} | Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss} | Val Acc: {val_acc}")
        print(
            f"Current LR: {optimizer.param_groups[0]['lr']} | Weight Decay: {current_wd}"
        )

        # Early Stopping Check
        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            # Deepcopy to strictly preserve best weights
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0

            # Save checkpoint
            ckpt_filename = f"{Config.MODEL_CHECKPOINT_PREFIX}_{fold_idx}.pth"
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": best_weights,
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                filename=ckpt_filename,
            )

            print(f"Saved best model for fold {fold_idx} with loss {best_loss}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Fold {fold_idx} finished. Best Val Loss: {best_loss}")
    return best_loss, best_acc
