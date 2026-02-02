import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, calculate_rmse, save_checkpoint
from library.dataset import TextDenoisingDataset, get_transforms
from library.model import CoSPResUNet


def pad_to_multiple(x, multiple=16):
    """
    Pads the input tensor (N, C, H, W) so that H and W are divisible by 'multiple'.
    Pads to the right and bottom using reflection padding.
    """
    h, w = x.shape[2], x.shape[3]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        # F.pad format: (padding_left, padding_right, padding_top, padding_bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    return x, pad_h, pad_w


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one training epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for noisy_imgs, clean_imgs in loader:
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        # Target is the noise residual
        noise_target = noisy_imgs - clean_imgs

        # Forward pass
        noise_pred = model(noisy_imgs)

        loss = criterion(noise_pred, noise_target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * noisy_imgs.size(0)
        count += noisy_imgs.size(0)

    return running_loss / count if count > 0 else 0.0


def validate_epoch(model, loader, device):
    """
    Performs validation on full images.
    Calculates RMSE between reconstructed clean image and ground truth.
    """
    model.eval()
    total_rmse = 0.0
    count = 0

    with torch.no_grad():
        for noisy_imgs, clean_imgs, _ in loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            # Pad input to be divisible by 16 (required for 4 pooling layers)
            padded_noisy, pad_h, pad_w = pad_to_multiple(noisy_imgs, multiple=16)

            # Forward pass (predict noise)
            noise_pred_padded = model(padded_noisy)

            # Crop back to original size
            h_padded, w_padded = noise_pred_padded.shape[2], noise_pred_padded.shape[3]
            noise_pred = noise_pred_padded[:, :, : h_padded - pad_h, : w_padded - pad_w]

            # Reconstruct clean image: Clean = Noisy - Noise
            clean_pred = noisy_imgs - noise_pred

            # Clip values to valid range [0, 1]
            clean_pred = torch.clamp(clean_pred, 0, 1)

            # Calculate RMSE
            # We iterate batch items to ensure correct calculation per image if batch > 1
            # (Though val batch size is typically 1 due to varying sizes)
            for i in range(noisy_imgs.size(0)):
                rmse = calculate_rmse(clean_pred[i], clean_imgs[i])
                total_rmse += rmse
                count += 1

    return total_rmse / count if count > 0 else 0.0


def train_model(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.EARLY_STOPPING_PATIENCE,
    num_workers=Config.NUM_WORKERS,
    max_train_samples=None,
):
    """
    Main training loop.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for optimizer.
        patience (int): Early stopping patience.
        num_workers (int): Number of dataloader workers.
        max_train_samples (int, optional): Limit training data size for debugging.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Initializing Datasets...")
    train_dataset = TextDenoisingDataset(
        metadata_path=Config.TRAIN_METADATA,
        mode="train",
        transform=get_transforms("train"),
    )

    val_dataset = TextDenoisingDataset(
        metadata_path=Config.VAL_METADATA, mode="val", transform=get_transforms("val")
    )

    # Debugging: Subset training data if requested
    if max_train_samples is not None and max_train_samples < len(train_dataset):
        print(f"Debugging: Subsetting training data to {max_train_samples} samples.")
        indices = list(range(max_train_samples))
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to maintain consistent stats
    )

    # Validation loader uses batch_size=1 because image sizes vary
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train samples (patches): {len(train_dataset)}")
    print(f"Val samples (images): {len(val_dataset)}")

    # --- Model Setup ---
    model = CoSPResUNet().to(device)

    # Residual learning: Minimize MSE between predicted noise and actual noise
    criterion = nn.MSELoss()

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # --- Training Loop ---
    best_rmse = float("inf")
    epochs_no_improve = 0

    print("\nStarting Training...")
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val RMSE':<12} | {'Time':<8}")
    print("-" * 45)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_rmse = validate_epoch(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Logging
        print(
            f"{epoch:<6} | {train_loss:<12.8f} | {val_rmse:<12.8f} | {elapsed:<8.1f}s"
        )

        # Checkpoint & Early Stopping
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_rmse)
            # print(f"  -> New best model saved! RMSE: {best_rmse:.8f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            print(f"Best Validation RMSE: {best_rmse:.8f}")
            break

    print("\nTraining Complete.")
    print(f"Best Validation RMSE: {best_rmse:.8f}")
