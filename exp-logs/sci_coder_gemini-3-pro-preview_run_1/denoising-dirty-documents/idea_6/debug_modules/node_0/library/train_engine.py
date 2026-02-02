import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    LEARNING_RATE,
    WEIGHT_DECAY,
    T_MAX,
    ETA_MIN,
    WORKING_DIR,
    SEED,
    EPOCHS,
)
from library.utils import seed_everything, worker_init_fn, rmse_score
from library.model import ShallowUNet
from library.dataset import DenoisingDataset, get_transforms


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (str): Device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate_one_epoch(model, dataloader, device):
    """
    Performs validation on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for validation data.
        device (str): Device to run validation on.

    Returns:
        float: Average RMSE score for the epoch.
    """
    model.eval()
    running_rmse = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Pad images to be divisible by 4 (due to 2 pooling layers in ShallowUNet)
            h, w = images.shape[2], images.shape[3]
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4

            if pad_h > 0 or pad_w > 0:
                images = F.pad(images, (0, pad_w, 0, pad_h))

            outputs = model(images)

            # Crop back to original size to match mask
            if pad_h > 0 or pad_w > 0:
                outputs = outputs[:, :, :h, :w]

            score = rmse_score(masks, outputs)
            running_rmse += score

    return running_rmse / len(dataloader)


def run_fold(fold_idx, train_imgs, train_masks, val_imgs, val_masks, epochs=EPOCHS):
    """
    Runs the training pipeline for a single fold.

    Args:
        fold_idx (int): Index of the current fold.
        train_imgs (np.array): Training images.
        train_masks (np.array): Training masks (clean images).
        val_imgs (np.array): Validation images.
        val_masks (np.array): Validation masks (clean images).
        epochs (int): Number of epochs to train.

    Returns:
        float: Best RMSE score achieved on validation set.
    """
    # Ensure reproducibility for this fold
    seed_everything(SEED + fold_idx)

    # Create Datasets
    # Train set gets geometric augmentation
    train_transform = get_transforms(mode="train")
    train_dataset = DenoisingDataset(train_imgs, train_masks, transform=train_transform)

    # Validation set gets only tensor conversion (no geometric augs)
    val_transform = get_transforms(mode="val")
    val_dataset = DenoisingDataset(val_imgs, val_masks, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=worker_init_fn,
    )

    # Validation batch size is 1 to handle varying image sizes safely
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Initialize Model
    model = ShallowUNet().to(DEVICE)

    # Optimizer and Scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=ETA_MIN
    )

    criterion = nn.MSELoss()

    best_rmse = float("inf")
    model_save_path = os.path.join(WORKING_DIR, f"model_fold_{fold_idx}.pth")

    print(f"Starting Fold {fold_idx} training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Step scheduler after the epoch
        scheduler.step()

        val_rmse = validate_one_epoch(model, val_loader, DEVICE)

        # Save best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), model_save_path)

        # Log progress periodically
        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            # Printing full precision for val_rmse as required
            print(
                f"Fold {fold_idx} | Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val RMSE: {val_rmse}"
            )

    print(f"Fold {fold_idx} Finished. Best RMSE: {best_rmse}")
    return best_rmse
