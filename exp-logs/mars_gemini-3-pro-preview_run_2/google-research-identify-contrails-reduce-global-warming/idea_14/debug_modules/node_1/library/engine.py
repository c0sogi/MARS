import os
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import AverageMeter, compute_intersection_union, rle_encode


def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler, scheduler=None
):
    """
    Performs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        with amp.autocast():
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.scale(optimizer).step()
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def valid_one_epoch(model, loader, criterion, device):
    """
    Performs validation and calculates Global Dice.
    """
    model.eval()
    losses = AverageMeter()

    total_intersection = 0.0
    total_union = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            logits = model(images)
            loss = criterion(logits, masks)

            losses.update(loss.item(), images.size(0))

            # Calculate intersection and union for Global Dice
            # We use the raw logits or probabilities depending on utils implementation
            # compute_intersection_union handles sigmoid application if needed
            inter, union = compute_intersection_union(
                logits, masks, threshold=Config.THRESHOLD
            )
            total_intersection += inter
            total_union += union

    # Global Dice = 2 * (Total Intersection) / (Total Union)
    # Add epsilon to avoid division by zero
    global_dice = (2.0 * total_intersection) / (total_union + 1e-6)

    return losses.avg, global_dice


def predict_tta(model, loader, device):
    """
    Performs inference with Test-Time Augmentation (TTA).
    Returns a list of RLE encoded strings.
    """
    model.eval()
    rle_predictions = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, dtype=torch.float32)

            # 1. Forward pass - Original
            pred_orig = torch.sigmoid(model(images))

            # 2. Forward pass - Horizontal Flip
            images_h = torch.flip(images, dims=[3])
            pred_h = torch.sigmoid(model(images_h))
            pred_h = torch.flip(pred_h, dims=[3])  # Flip back

            # 3. Forward pass - Vertical Flip
            images_v = torch.flip(images, dims=[2])
            pred_v = torch.sigmoid(model(images_v))
            pred_v = torch.flip(pred_v, dims=[2])  # Flip back

            # 4. Forward pass - Rotate 180 (equivalent to H+V flip)
            images_rot = torch.rot90(images, k=2, dims=[2, 3])
            pred_rot = torch.sigmoid(model(images_rot))
            pred_rot = torch.rot90(pred_rot, k=-2, dims=[2, 3])  # Rotate back

            # Average predictions
            avg_pred = (pred_orig + pred_h + pred_v + pred_rot) / 4.0

            # Convert to numpy and encode
            avg_pred = avg_pred.cpu().numpy()

            for i in range(avg_pred.shape[0]):
                # Extract single mask: (1, H, W) -> (H, W)
                mask = avg_pred[i, 0, :, :]
                rle = rle_encode(mask, threshold=Config.THRESHOLD)
                rle_predictions.append(rle)

    return rle_predictions


def train_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    device,
    epochs=Config.EPOCHS,
    patience=5,
    save_path=None,
):
    """
    Main training loop with Early Stopping.
    """
    if save_path is None:
        save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    scaler = amp.GradScaler()

    # Setup scheduler (CosineAnnealingLR is standard for this task)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * epochs, eta_min=Config.MIN_LR
    )

    best_dice = -1.0
    early_stop_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, scheduler
        )
        val_loss, val_dice = valid_one_epoch(model, valid_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.18f}")
        print(f"  Valid Loss: {val_loss:.18f}")
        print(f"  Global Dice: {val_dice:.18f}")

        # Checkpointing and Early Stopping
        if val_dice > best_dice:
            print(
                f"  Score Improved ({best_dice:.6f} --> {val_dice:.6f}). Saving model to {save_path}"
            )
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(
                f"  Score did not improve. Early stopping counter: {early_stop_counter}/{patience}"
            )

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Global Dice: {best_dice:.18f}")


def make_submission(model, loader, output_path, device):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")

    # Ensure model is in eval mode
    model.eval()

    # Get predictions (list of RLE strings)
    # Note: predict_tta iterates sequentially, preserving order
    predictions = predict_tta(model, loader, device)

    # Get record_ids from the dataset dataframe
    # The DataLoader preserves the order of the underlying dataset
    record_ids = loader.dataset.df["record_id"].values

    # Verify lengths match
    if len(predictions) != len(record_ids):
        raise ValueError(
            f"Mismatch: {len(predictions)} predictions vs {len(record_ids)} records."
        )

    # Create DataFrame
    submission_df = pd.DataFrame(
        {"record_id": record_ids, "encoded_pixels": predictions}
    )

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
