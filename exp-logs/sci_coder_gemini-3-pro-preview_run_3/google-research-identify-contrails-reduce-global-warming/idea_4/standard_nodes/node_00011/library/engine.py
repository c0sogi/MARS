import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import rle_encode


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    Handles Deep Supervision by summing losses from all decoder outputs.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        # In training mode, the U-Net++ model returns a list of logits
        # [logit_0_4, logit_0_3, logit_0_2, logit_0_1] for deep supervision.
        outputs = model(images)

        # Deep Supervision Loss Aggregation
        # We sum the loss from each output head against the ground truth.
        loss = 0.0
        if isinstance(outputs, list):
            for output in outputs:
                loss += criterion(output, masks)
        else:
            # Fallback if model structure changes to single output
            loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        # Accumulate loss (multiply by batch size to get total, then divide later)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes the Global Dice Coefficient.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    # Accumulators for Global Dice (Intersection and Union over all pixels in dataset)
    total_intersection = 0.0
    total_union = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            batch_size = images.size(0)

            # Forward pass
            # In eval mode, the model returns only the final output tensor.
            logits = model(images)

            # Compute Loss
            loss = criterion(logits, masks)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Compute Dice Statistics
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Flatten to treat as a single set of pixels
            preds_flat = preds.view(-1)
            targets_flat = masks.view(-1)

            intersection = (preds_flat * targets_flat).sum().item()
            union = preds_flat.sum().item() + targets_flat.sum().item()

            total_intersection += intersection
            total_union += union

    avg_loss = running_loss / dataset_size

    # Calculate Global Dice
    smooth = 1e-6
    global_dice = (2.0 * total_intersection + smooth) / (total_union + smooth)

    return avg_loss, global_dice


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs=Config.EPOCHS,
    checkpoint_manager=None,
    patience=10,
):
    """
    Main training loop.
    Handles epoch iteration, validation, scheduling, checkpointing, and early stopping.
    """
    best_dice = -1.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        # --- Training ---
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # --- Validation ---
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # --- Scheduler Step ---
        if scheduler is not None:
            # CosineAnnealingLR steps per epoch
            scheduler.step()

        # --- Logging ---
        # Printing full precision metrics as requested
        print(
            f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Dice: {val_dice}"
        )

        # --- Checkpointing ---
        if checkpoint_manager is not None:
            # Save model if it's among the top K
            checkpoint_manager.save(model, epoch, val_dice)

        # --- Early Stopping ---
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. Best Val Dice: {best_dice}"
                )
                break


def predict_and_submit(model, test_loader, device, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    results = []

    print("Generating predictions for submission...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            record_ids = batch["record_id"]

            # Inference
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Thresholding
            preds = (probs > 0.5).float().cpu().numpy()

            # Encode each image in the batch
            for i, record_id in enumerate(record_ids):
                # preds shape is (B, 1, H, W)
                # Extract single mask: (1, H, W) -> (H, W)
                mask = preds[i, 0, :, :]

                # Run-Length Encoding
                encoded_pixels = rle_encode(mask)

                results.append(
                    {"record_id": record_id, "encoded_pixels": encoded_pixels}
                )

    # Create DataFrame and save
    df = pd.DataFrame(results)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
