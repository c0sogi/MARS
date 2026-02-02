import os
import torch
import torch.nn as nn
import numpy as np
from library import config, utils


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Executes one training epoch using MSE Loss.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Training dataloader.
        optimizer (Optimizer): The optimizer.
        device (str): Device to run on.
        epoch (int): Current epoch number.

    Returns:
        float: Average MSE loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    criterion = nn.MSELoss()

    for i, data in enumerate(dataloader):
        # dataset.py in train mode returns (noisy, clean)
        noisy, clean = data
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        outputs = model(noisy)
        loss = criterion(outputs, clean)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set and returns RMSE.
    Calculates error on the original unpadded image dimensions to ensure accuracy.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Validation dataloader.
        device (str): Device to run on.

    Returns:
        float: RMSE value.
    """
    model.eval()
    total_sse = 0.0
    total_pixels = 0

    with torch.no_grad():
        for data in dataloader:
            # dataset.py in val mode returns (noisy, clean, meta)
            if len(data) == 3:
                noisy, clean, meta = data
            else:
                continue

            noisy = noisy.to(device)
            clean = clean.to(device)

            # Forward pass
            outputs = model(noisy)

            # Unpad for accurate metric calculation
            # Batch size is 1 for validation as per dataset.py
            h_orig = meta["orig_h"].item()
            w_orig = meta["orig_w"].item()

            # Slice the valid region (padding is on bottom/right)
            pred_valid = outputs[0, 0, :h_orig, :w_orig]
            target_valid = clean[0, 0, :h_orig, :w_orig]

            # Sum of Squared Errors
            sse = torch.sum((pred_valid - target_valid) ** 2).item()
            total_sse += sse
            total_pixels += h_orig * w_orig

    if total_pixels == 0:
        return 0.0

    mse = total_sse / total_pixels
    rmse = np.sqrt(mse)
    return rmse


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    save_path,
    patience=None,
):
    """
    Runs the full training pipeline with validation, scheduling, and early stopping.

    Args:
        model (nn.Module): The neural network.
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (str): Device.
        num_epochs (int): Total epochs.
        save_path (str): Path to save the best model.
        patience (int, optional): Early stopping patience.

    Returns:
        float: Best validation RMSE achieved.
    """
    best_rmse = float("inf")
    patience_counter = 0

    # Ensure save directory exists
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_rmse = evaluate(model, val_loader, device)

        if scheduler:
            scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train MSE: {train_loss:.8f} | Val RMSE: {val_rmse:.20f}"
        )

        # Save best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        # Early stopping
        if patience is not None and patience_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best RMSE: {best_rmse:.20f}"
            )
            break

    print(f"Training complete. Best Val RMSE: {best_rmse:.20f}")
    return best_rmse


def predict(model, test_loader, device, output_path=config.SUBMISSION_FILE_PATH):
    """
    Generates predictions using D4 Test-Time Augmentation (TTA) and saves to CSV.

    Args:
        model (nn.Module): The trained neural network.
        test_loader (DataLoader): Test dataloader.
        device (str): Device.
        output_path (str): Path to save the submission CSV.
    """
    model.eval()
    predictions = {}

    print("Generating predictions with D4 TTA...")

    with torch.no_grad():
        for data in test_loader:
            # dataset.py in test mode returns (noisy, meta)
            if len(data) == 2:
                noisy, meta = data
            else:
                continue

            img_id = meta["id"][0]
            orig_h = meta["orig_h"].item()
            orig_w = meta["orig_w"].item()

            x = noisy.to(device)  # (1, 1, H, W)

            # --- D4 TTA (8 views) ---
            # Group 1: No Transpose (H, W) -> [Original, Rot180, Flip, Flip+Rot180]
            x_rot180 = torch.rot90(x, 2, [2, 3])
            x_flip = torch.flip(x, [3])
            x_flip_rot180 = torch.rot90(x_flip, 2, [2, 3])

            batch1 = torch.cat([x, x_rot180, x_flip, x_flip_rot180], dim=0)
            out1 = model(batch1)

            # Inverse Group 1
            y1 = out1[0:1]
            y2 = torch.rot90(out1[1:2], 2, [2, 3])  # Inv rot180 is rot180
            y3 = torch.flip(out1[2:3], [3])  # Inv flip is flip
            y4 = torch.flip(
                torch.rot90(out1[3:4], 2, [2, 3]), [3]
            )  # Inv (Flip->Rot180) is (Rot180->Flip)

            # Group 2: Transpose (W, H) -> [Rot90, Rot270, Flip+Rot90, Flip+Rot270]
            x_rot90 = torch.rot90(x, 1, [2, 3])
            x_rot270 = torch.rot90(x, 3, [2, 3])
            x_flip_rot90 = torch.rot90(x_flip, 1, [2, 3])
            x_flip_rot270 = torch.rot90(x_flip, 3, [2, 3])

            batch2 = torch.cat([x_rot90, x_rot270, x_flip_rot90, x_flip_rot270], dim=0)
            out2 = model(batch2)

            # Inverse Group 2
            # Inv Rot90 is Rot270 (k=3)
            y5 = torch.rot90(out2[0:1], 3, [2, 3])
            # Inv Rot270 is Rot90 (k=1)
            y6 = torch.rot90(out2[1:2], 1, [2, 3])
            # Inv (Flip->Rot90) is (Rot270->Flip) -> Flip(Rot90(y, 3))
            y7 = torch.flip(torch.rot90(out2[2:3], 3, [2, 3]), [3])
            # Inv (Flip->Rot270) is (Rot90->Flip) -> Flip(Rot90(y, 1))
            y8 = torch.flip(torch.rot90(out2[3:4], 1, [2, 3]), [3])

            # Average all 8 views
            y_avg = (y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8) / 8.0

            # Extract valid region (unpad)
            pred = y_avg[0, 0, :orig_h, :orig_w].cpu().numpy()
            predictions[img_id] = pred

    utils.create_submission(predictions, output_path)
    print(f"Predictions saved to {output_path}")
