import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.model import ResUNet, predict_tiled
from library.utils import set_seed


def train_one_epoch(model, dataloader, optimizer, criterion, device, max_batches=None):
    """
    Performs one epoch of training using Global Residual Learning.
    The model predicts the noise, and the loss is calculated against (Input - Clean).

    Args:
        model: The PyTorch model.
        dataloader: Training DataLoader yielding (noisy_patch, clean_patch).
        optimizer: The optimizer.
        criterion: The loss function (MSE).
        device: 'cuda' or 'cpu'.
        max_batches (int, optional): Limit the number of batches for debugging.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for i, (noisy, clean) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        # Global Residual Learning: The model predicts the noise map
        pred_noise = model(noisy)

        # Target Noise = Input Noisy Image - Ground Truth Clean Image
        target_noise = noisy - clean

        loss = criterion(pred_noise, target_noise)
        loss.backward()
        optimizer.step()

        # Accumulate loss weighted by batch size
        running_loss += loss.item() * noisy.size(0)
        count += noisy.size(0)

    return running_loss / count if count > 0 else 0.0


def evaluate(model, val_data, device, max_samples=None):
    """
    Evaluates the model on the validation set using RMSE.
    Performs tiled inference on full-resolution images.

    Args:
        model: The PyTorch model.
        val_data: List of dictionaries containing 'noisy' and 'clean' numpy arrays.
        device: 'cuda' or 'cpu'.
        max_samples (int, optional): Limit the number of validation samples for debugging.

    Returns:
        float: Root Mean Squared Error (RMSE) over all pixels.
    """
    model.eval()
    val_mse = 0.0
    total_pixels = 0
    count = 0

    for item in val_data:
        if max_samples is not None and count >= max_samples:
            break

        noisy_np = item["noisy"]
        clean_np = item["clean"]

        # Prepare tensors: (1, 1, H, W)
        # predict_tiled expects input shape (C, H, W)
        noisy_t = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0).float()
        clean_t = (
            torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0).float().to(device)
        )

        with torch.no_grad():
            # Squeeze batch dim to pass (C, H, W) to predict_tiled
            # predict_tiled returns (C, H, W)
            pred_clean = predict_tiled(model, noisy_t.squeeze(0), device=device)

            # Add batch dim back for comparison: (1, C, H, W)
            pred_clean = pred_clean.unsqueeze(0)

        # Calculate Squared Error
        diff = (pred_clean - clean_t) ** 2
        val_mse += diff.sum().item()
        total_pixels += diff.numel()
        count += 1

    if total_pixels == 0:
        return float("inf")

    # Calculate RMSE
    rmse = np.sqrt(val_mse / total_pixels)
    return rmse


def train_engine(
    train_loader,
    val_data,
    epochs=50,
    lr=1e-4,
    device="cuda",
    save_path="./working/model.pth",
    patience=5,
    max_train_batches=None,
    max_val_samples=None,
    seed=42,
):
    """
    Main training loop with validation and early stopping.

    Args:
        train_loader: DataLoader for training patches.
        val_data: List of validation images.
        epochs (int): Maximum number of epochs.
        lr (float): Learning rate.
        device (str): Device to train on.
        save_path (str): Path to save the best model.
        patience (int): Early stopping patience.
        max_train_batches (int, optional): Limit training steps per epoch.
        max_val_samples (int, optional): Limit validation steps per epoch.
        seed (int): Random seed.

    Returns:
        float: Best Validation RMSE achieved.
    """
    set_seed(seed)

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = ResUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_batches=max_train_batches,
        )

        scheduler.step()

        val_rmse = evaluate(model, val_data, device, max_samples=max_val_samples)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val RMSE: {val_rmse}"
        )

        # Checkpoint and Early Stopping
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best Val RMSE: {best_rmse}")
    return best_rmse
