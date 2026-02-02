import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import HubmapDataset
from library.model import UnetPlusPlus
from library.losses import DeepSupervisionLoss
from library.utils import set_seed


class GPUAugmentor(nn.Module):
    """
    Applies data augmentation directly on GPU tensors to maximize throughput.
    Includes: Random Horizontal/Vertical Flip, Random Rotate90, and ColorJitter.
    """

    def __init__(self):
        super().__init__()
        # Photometric augmentations (applied only to images)
        # Note: Input images are [0, 1] float tensors.
        self.color_jitter = T.ColorJitter(
            brightness=Config.SCALE_LIMIT,
            contrast=Config.SCALE_LIMIT,
            saturation=Config.SCALE_LIMIT,
            hue=0.05,
        )

    def forward(self, images, masks):
        """
        Args:
            images (torch.Tensor): (B, 3, H, W)
            masks (torch.Tensor): (B, 1, H, W)
        """
        B = images.shape[0]

        # 1. Photometric Augmentation (Randomly applied batch-wise for speed)
        # We apply jitter with probability AUG_PROB
        if torch.rand(1).item() < Config.AUG_PROB:
            images = self.color_jitter(images)

        # 2. Geometric Augmentations (Must apply consistently to image and mask)

        # Random Horizontal Flip
        if torch.rand(1).item() < 0.5:
            images = torch.flip(images, [3])
            masks = torch.flip(masks, [3])

        # Random Vertical Flip
        if torch.rand(1).item() < 0.5:
            images = torch.flip(images, [2])
            masks = torch.flip(masks, [2])

        # Random Rotate 90 (0, 90, 180, 270 degrees)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            images = torch.rot90(images, k, [2, 3])
            masks = torch.rot90(masks, k, [2, 3])

        return images, masks


def train_one_epoch(
    model, dataloader, optimizer, scaler, loss_fn, device, augmentor=None
):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Apply GPU Augmentations
        if augmentor is not None:
            with torch.no_grad():
                images, masks = augmentor(images, masks)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with autocast():
            outputs = model(images)
            loss = loss_fn(outputs, masks)

        # Backward Pass with Scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, loss_fn, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Dice Coefficient.
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    dataset_size = 0

    # Dice calculation helper
    smooth = 1e-6

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Mixed Precision Inference
            with autocast():
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            # Deep Supervision returns a list. Use the first (highest res) output for metrics.
            if isinstance(outputs, list):
                preds = outputs[0]
            else:
                preds = outputs

            # Compute Dice
            preds_prob = torch.sigmoid(preds)
            preds_bin = (preds_prob > Config.THRESHOLD).float()

            intersection = (preds_bin * masks).sum(dim=(1, 2, 3))
            union = preds_bin.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + smooth) / (union + smooth)

            running_loss += loss.item() * images.size(0)
            running_dice += dice.sum().item()
            dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    epoch_dice = running_dice / dataset_size
    return epoch_loss, epoch_dice


def train_model(
    num_epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    samples_per_epoch=None,
    patience=5,
):
    """
    Main function to train the model.

    Args:
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        samples_per_epoch (int, optional): Limit training samples per epoch for debugging.
        patience (int): Early stopping patience.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Training on device: {device}")

    # 1. Datasets and Dataloaders
    # Note: transform=None because we use GPUAugmentor
    train_dataset = HubmapDataset(
        mode="train", transform=None, samples_per_epoch=samples_per_epoch
    )
    val_dataset = HubmapDataset(mode="val", transform=None)

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

    # 2. Model, Optimizer, Loss
    model = UnetPlusPlus(
        backbone_name=Config.BACKBONE,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    criterion = DeepSupervisionLoss(weights=Config.LOSS_WEIGHTS)
    scaler = GradScaler()
    augmentor = GPUAugmentor().to(device)

    # 3. Training Loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = 0.0
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, augmentor
        )

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Print Metrics (Full Precision)
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Dice: {val_dice}")
        print(f"LR: {current_lr}")

        # Deep Copy Model if Best
        if val_dice > best_dice:
            best_dice = val_dice
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            print(f"New best model saved to {Config.CHECKPOINT_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early Stopping
        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {epochs_no_improve} epochs of no improvement."
            )
            break

        print("-" * 30)

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Dice: {best_dice}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model
