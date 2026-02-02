import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import AverageMeter, rle_encode
from library.loss import FocalBatchDiceLoss


def train_one_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()
    criterion = FocalBatchDiceLoss(gamma=Config.FOCAL_GAMMA)

    for images, masks in loader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using Global Dice Coefficient.
    """
    model.eval()
    loss_meter = AverageMeter()
    criterion = FocalBatchDiceLoss(gamma=Config.FOCAL_GAMMA)

    # Accumulators for Global Dice
    total_intersection = 0.0
    total_union = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            logits = model(images)
            loss = criterion(logits, masks)
            loss_meter.update(loss.item(), images.size(0))

            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Flatten for global calculation
            preds_flat = preds.view(-1)
            targets_flat = masks.view(-1)

            intersection = (preds_flat * targets_flat).sum().item()
            union = preds_flat.sum().item() + targets_flat.sum().item()

            total_intersection += intersection
            total_union += union

    # Compute Global Dice
    # Formula: 2 * |X n Y| / (|X| + |Y|)
    epsilon = 1e-6
    global_dice = (2.0 * total_intersection + epsilon) / (total_union + epsilon)

    return global_dice, loss_meter.avg


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=Config.EPOCHS,
    patience=5,
):
    """
    Runs the full training loop with Early Stopping and Model Checkpointing.
    """
    best_dice = 0.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on device: {device}")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_dice, val_loss = validate(model, val_loader, device)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        # Print metrics
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Dice: {val_dice:.12f}"
        )

        # Checkpointing
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  New best model saved! (Dice: {best_dice:.12f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"Training complete. Best Global Dice: {best_dice:.12f}")


def inference(model, loader, device):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    Saves the result to submission.csv.
    """
    model.eval()

    # Load metadata to get record_ids
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    record_ids = test_df["record_id"].astype(str).tolist()

    encoded_pixels = []

    print("Starting inference with TTA...")

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device, dtype=torch.float32)

            # --- Test-Time Augmentation (TTA) ---
            # 1. Original
            logits_1 = model(images)
            probs_1 = torch.sigmoid(logits_1)

            # 2. Horizontal Flip
            images_h = torch.flip(images, dims=[3])
            logits_2 = model(images_h)
            probs_2 = torch.flip(torch.sigmoid(logits_2), dims=[3])

            # 3. Vertical Flip
            images_v = torch.flip(images, dims=[2])
            logits_3 = model(images_v)
            probs_3 = torch.flip(torch.sigmoid(logits_3), dims=[2])

            # 4. 180 Degree Rotation
            images_rot = torch.rot90(images, k=2, dims=[2, 3])
            logits_4 = model(images_rot)
            probs_4 = torch.rot90(torch.sigmoid(logits_4), k=-2, dims=[2, 3])

            # Average predictions
            avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0

            # Thresholding
            preds = (avg_probs > 0.5).cpu().numpy()

            # Encode batch
            for j in range(preds.shape[0]):
                mask = preds[j, 0, :, :]  # (H, W)
                rle = rle_encode(mask)
                encoded_pixels.append(rle)

    # Create submission DataFrame
    submission = pd.DataFrame(
        {"record_id": record_ids, "encoded_pixels": encoded_pixels}
    )

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
