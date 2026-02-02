import os
import time
import torch
import numpy as np
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model returns logits (no sigmoid applied in model forward)
        outputs = model(images)

        # Compute loss (HybridBCEDiceLoss expects logits)
        loss = criterion(outputs, masks)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Aggregate loss
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, dataloader, criterion, device, threshold=0.5):
    """
    Performs one epoch of validation.
    Computes loss and Global Dice Coefficient.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    # Accumulators for Global Dice
    total_intersection = 0.0
    total_union = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, masks)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Compute metrics
            # Apply sigmoid to logits for predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            # Flatten for global calculation over the whole validation set
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            intersection = (preds_flat * masks_flat).sum().item()
            union = preds_flat.sum().item() + masks_flat.sum().item()

            total_intersection += intersection
            total_union += union

    epoch_loss = running_loss / dataset_size

    # Global Dice: 2 * |X n Y| / (|X| + |Y|)
    epsilon = 1e-6
    global_dice = (2.0 * total_intersection) / (total_union + epsilon)

    return epoch_loss, global_dice


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs,
    patience=5,
):
    """
    Orchestrates the training process with logging, checkpointing, and early stopping.
    """
    best_dice = 0.0
    patience_counter = 0
    best_model_path = Config.BEST_MODEL_PATH

    # Ensure working directory exists
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        val_loss, val_dice = valid_one_epoch(
            model, val_loader, criterion, device, threshold=Config.THRESHOLD
        )

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - start_time

        # Logging
        print(f"Epoch {epoch+1}/{epochs} - Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        # Print full precision for validation metric
        print(f"  Val Global Dice: {val_dice}")

        # Checkpointing and Early Stopping
        if val_dice > best_dice:
            print(f"  Validation Dice Improved ({best_dice} ---> {val_dice})")
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"  Model saved to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    print(f"Best Validation Dice: {best_dice}")

    # Load best model weights before returning
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model
