import os
import torch
import torch.optim as optim
import numpy as np
from library.config import get_config
from library.utils import set_seed
from library.loss import ContrailLoss
from library.model import HRNetSegmenter
from library.dataset import get_dataloader


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        loss_fn (nn.Module): The loss function.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Compute loss
        loss = loss_fn(logits, masks)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, loss_fn, device):
    """
    Evaluates the model on the validation set.
    Computes the average loss and the Global Dice Coefficient.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Validation data loader.
        loss_fn (nn.Module): The loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (average_loss, global_dice_score)
    """
    model.eval()
    running_loss = 0.0

    # Variables for Global Dice calculation
    intersection_sum = 0.0
    union_sum = 0.0
    smooth = 1e-6
    threshold = 0.5

    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)

            # Compute Loss
            loss = loss_fn(logits, masks)
            running_loss += loss.item()

            # Compute Global Dice components
            probs = torch.sigmoid(logits)
            pred_mask = (probs > threshold).float()

            # Flatten for calculation
            pred_flat = pred_mask.contiguous().view(-1)
            true_flat = masks.contiguous().view(-1).float()

            intersection_sum += (pred_flat * true_flat).sum().item()
            union_sum += pred_flat.sum().item() + true_flat.sum().item()

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Compute Global Dice
    global_dice = (2.0 * intersection_sum + smooth) / (union_sum + smooth)

    return avg_loss, global_dice


def train_model(debug=False):
    """
    Main training loop.

    Args:
        debug (bool): If True, runs with debug configuration (fewer epochs, smaller dataset).
    """
    # 1. Load Configuration
    Config = get_config(debug=debug)

    # 2. Set Seeds
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Initialize DataLoaders
    print("Initializing DataLoaders...")
    train_loader = get_dataloader(
        mode="train",
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=debug,
    )
    val_loader = get_dataloader(
        mode="validation",
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=debug,
    )

    # 4. Initialize Model
    print(f"Initializing Model: {Config.MODEL_NAME}...")
    model = HRNetSegmenter(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # 5. Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    loss_fn = ContrailLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # 6. Scheduler (ReduceLROnPlateau monitoring Validation Dice)
    # Mode is 'max' because we want to maximize Dice score
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # 7. Training Loop
    best_dice = 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )

        # Validate
        val_loss, val_dice = validate(model, val_loader, loss_fn, device)

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Dice: {val_dice}"
        )

        # Scheduler Step
        scheduler.step(val_dice)

        # Checkpointing & Early Stopping
        if val_dice > best_dice:
            print(
                f"Validation Dice improved from {best_dice} to {val_dice}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in Validation Dice for {epochs_no_improve} epochs.")

        if epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered after {epochs_no_improve} epochs with no improvement."
            )
            break

    print(f"Training complete. Best Validation Dice: {best_dice}")
    print(f"Best model saved to: {best_model_path}")
