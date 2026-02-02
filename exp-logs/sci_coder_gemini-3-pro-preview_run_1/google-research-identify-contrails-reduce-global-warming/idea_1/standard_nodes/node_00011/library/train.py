import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import ContrailDataset
from library.model import SimpleUNet
from library.utils import AverageMeter, GlobalDiceMetric, dice_coefficient


class ContrailLoss(nn.Module):
    """
    Combined Loss: Weighted sum of Focal Loss and Dice Loss.
    Addresses class imbalance where background dominates.
    """

    def __init__(
        self, focal_weight=1.0, dice_weight=1.0, alpha=0.5, gamma=2.0, smooth=1e-6
    ):
        super(ContrailLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Clamp predictions for numerical stability
        y_pred = torch.clamp(y_pred, 1e-7, 1.0 - 1e-7)

        # Focal Loss
        # Calculate BCE term
        bce = -y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred)

        # Calculate weighting factors
        # pt = p if y=1 else 1-p
        pt = torch.where(y_true == 1, y_pred, 1 - y_pred)
        # alpha_t = alpha if y=1 else 1-alpha
        alpha_t = torch.where(y_true == 1, self.alpha, 1 - self.alpha)

        focal_loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        focal_loss = focal_loss.mean()

        # Dice Loss
        y_pred_f = y_pred.view(-1)
        y_true_f = y_true.view(-1)
        intersection = (y_pred_f * y_true_f).sum()
        union = y_pred_f.sum() + y_true_f.sum()
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Handles the training loop for a single epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Handles the validation loop. Computes loss and Global Dice Score.
    """
    model.eval()
    losses = AverageMeter()
    dice_metric = GlobalDiceMetric()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            losses.update(loss.item(), images.size(0))

            # Update global dice metric with thresholded predictions
            dice_metric.update(outputs, masks, threshold=Config.THRESHOLD)

    global_dice = dice_metric.compute()
    return losses.avg, global_dice


def train_model(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    debug=False,
    max_samples=None,
    patience=3,
    train_transform=None,
):
    """
    Main training function.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate for Adam optimizer.
        weight_decay (float): Weight decay for Adam optimizer.
        debug (bool): If True, limits dataset size for quick debugging.
        max_samples (int): Overrides debug limit if provided.
        patience (int): Early stopping patience.
    """
    # Set reproducibility
    Config.set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Starting training on device: {device}")

    # --------------------------------------------------------------------------
    # 1. Data Loading
    # --------------------------------------------------------------------------
    if debug and max_samples is None:
        max_samples = 100
        print("Debug mode: Limiting dataset to 100 samples.")

    train_dataset = ContrailDataset(
        split="train", max_samples=max_samples, transform=train_transform
    )
    val_dataset = ContrailDataset(split="validation", max_samples=max_samples)

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

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # --------------------------------------------------------------------------
    # 2. Model & Optimization
    # --------------------------------------------------------------------------
    model = SimpleUNet(in_channels=Config.NUM_BANDS, out_channels=1).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Use Config weights and alpha=0.5 (Cite solution_lesson_node_00006, solution_lesson_node_00010)
    criterion = ContrailLoss(
        focal_weight=Config.BCE_WEIGHT,
        dice_weight=Config.DICE_WEIGHT,
        alpha=0.5,
    )

    # Scheduler (Cite solution_lesson_node_00006)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    # --------------------------------------------------------------------------
    # 3. Training Loop
    # --------------------------------------------------------------------------
    best_dice = 0.0
    early_stop_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step(val_dice)

        # Logging
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Global Dice: {val_dice}")

        # Checkpoint & Early Stopping
        if val_dice > best_dice:
            print(
                f"Validation Dice improved from {best_dice} to {val_dice}. Saving checkpoint."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(
                f"No improvement. Early stopping counter: {early_stop_counter}/{patience}"
            )

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

        print("-" * 30)

    print(f"Training complete. Best Validation Dice: {best_dice}")
    return best_dice
