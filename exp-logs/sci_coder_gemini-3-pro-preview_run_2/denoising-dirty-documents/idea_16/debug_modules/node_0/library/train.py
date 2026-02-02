import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import (
    DEVICE,
    NUM_WORKERS,
    SEED,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    PATCH_SIZE,
    OVERLAP_RATIO,
    CHECKPOINT_DIR,
    DEBUG_SAMPLE_SIZE,
)
from library.utils import seed_everything, save_checkpoint, calculate_rmse
from library.model import ICResUNet
from library.dataset import DenoisingDataset


def predict_tiled(model, noisy_tensor, patch_size=PATCH_SIZE, overlap=OVERLAP_RATIO):
    """
    Performs sliding window inference on a large image tensor.

    Args:
        model: The trained neural network.
        noisy_tensor: Input tensor of shape (C, H, W).
        patch_size: Size of the square patch.
        overlap: Overlap ratio (0.0 to 1.0).

    Returns:
        clean_pred: Denoised image tensor of shape (C, H, W).
    """
    C, H, W = noisy_tensor.shape
    stride = int(patch_size * (1 - overlap))

    # Buffers for accumulation
    noise_sum = torch.zeros((C, H, W), device=noisy_tensor.device)
    count_map = torch.zeros((C, H, W), device=noisy_tensor.device)

    # Calculate starting coordinates for patches
    y_starts = []
    y = 0
    while y + patch_size < H:
        y_starts.append(y)
        y += stride
    y_starts.append(H - patch_size)  # Ensure last patch covers the edge

    x_starts = []
    x = 0
    while x + patch_size < W:
        x_starts.append(x)
        x += stride
    x_starts.append(W - patch_size)  # Ensure last patch covers the edge

    # Sliding window loop
    for y in y_starts:
        for x in x_starts:
            # Extract patch
            patch = noisy_tensor[:, y : y + patch_size, x : x + patch_size]
            patch = patch.unsqueeze(0)  # Add batch dim: (1, C, H, W)

            # Predict noise
            with torch.no_grad():
                pred_noise_patch = model(patch)

            pred_noise_patch = pred_noise_patch.squeeze(0)  # Remove batch dim

            # Accumulate
            noise_sum[:, y : y + patch_size, x : x + patch_size] += pred_noise_patch
            count_map[:, y : y + patch_size, x : x + patch_size] += 1.0

    # Average the predictions
    avg_noise = noise_sum / count_map

    # Calculate clean image: Clean = Noisy - Noise
    clean_pred = noisy_tensor - avg_noise

    # Clamp to valid range
    return torch.clamp(clean_pred, 0.0, 1.0)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for noisy, clean in dataloader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        # Target is the noise residual
        target_noise = noisy - clean

        optimizer.zero_grad()

        # Forward pass
        pred_noise = model(noisy)

        # Loss calculation
        loss = criterion(pred_noise, target_noise)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * noisy.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set using tiled inference.
    Returns the average RMSE.
    """
    model.eval()
    rmse_scores = []

    # Disable gradient calculation for validation
    with torch.no_grad():
        for noisy, clean, _ in dataloader:
            # Validation dataloader returns single items with batch dim 1 usually,
            # but our dataset returns (1, H, W). DataLoader stacks them to (B, 1, H, W).
            # We process image by image for tiled inference logic simplicity.

            for i in range(noisy.size(0)):
                img_noisy = noisy[i].to(device)  # (1, H, W)
                img_clean = clean[i].to(device)  # (1, H, W)

                # Predict
                img_pred = predict_tiled(model, img_noisy)

                # Calculate RMSE
                score = calculate_rmse(img_clean, img_pred)
                rmse_scores.append(score)

    return np.mean(rmse_scores)


def run_training(debug=False, epochs=NUM_EPOCHS):
    """
    Main training function.
    """
    seed_everything(SEED)

    # 1. Data Preparation
    train_dataset = DenoisingDataset(mode="train")
    val_dataset = DenoisingDataset(mode="val")

    if debug:
        print(f"Debug mode: Truncating datasets to {DEBUG_SAMPLE_SIZE} samples.")
        train_dataset.data = train_dataset.data[:DEBUG_SAMPLE_SIZE]
        val_dataset.data = val_dataset.data[:DEBUG_SAMPLE_SIZE]

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Batch size 1 for validation to handle varying image sizes easily if needed,
    # though dataset pads/resizes? No, dataset returns full images.
    # Batching full images requires them to be same size.
    # EDA showed varying heights. So we must use batch_size=1 for validation.
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS
    )

    # 2. Model Initialization
    model = ICResUNet().to(DEVICE)

    # 3. Optimization
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Training Loop
    best_rmse = float("inf")
    patience_counter = 0

    print(f"Starting training on {DEVICE} for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_rmse = validate(model, val_loader, DEVICE)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val RMSE: {val_rmse:.8f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, train_loss, filename="best_model.pth"
            )
            # Also save latest for safety
            save_checkpoint(
                model, optimizer, epoch, train_loss, filename="latest_model.pth"
            )
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch}. Best Val RMSE: {best_rmse:.8f}"
            )
            break

    print(f"Training complete. Best Validation RMSE: {best_rmse:.8f}")
