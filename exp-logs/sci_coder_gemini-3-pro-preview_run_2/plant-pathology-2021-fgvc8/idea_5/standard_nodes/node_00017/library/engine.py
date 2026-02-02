import torch
import torch.nn as nn
import numpy as np
import sys
from library.config import Config
from library.utils import calculate_f1_score, save_checkpoint


def train_one_epoch(model, loader, optimizer, device, scaler, epoch):
    """
    Trains the model for one epoch using Gradient Accumulation and Mixed Precision.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Define Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Zero gradients at the start of the epoch
    optimizer.zero_grad()

    for step, (images, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, targets)
            # Normalize loss for gradient accumulation
            loss = loss / Config.GRAD_ACCUM_STEPS

        # Backward Pass
        scaler.scale(loss).backward()

        # Accumulate loss for reporting (multiply back to get actual batch loss)
        running_loss += loss.item() * Config.GRAD_ACCUM_STEPS * batch_size
        dataset_size += batch_size

        # Optimizer Step (Gradient Accumulation)
        if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()

            # Zero Gradients
            optimizer.zero_grad()

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and F1 score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = images.size(0)

            # Forward pass (Mixed Precision is generally safe/faster for inference too)
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store predictions and targets for metric calculation
            # Apply sigmoid to convert logits to probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / dataset_size

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate F1 Score
    val_score = calculate_f1_score(all_targets, all_preds)

    return val_loss, val_score


def run_training(
    model, train_loader, val_loader, optimizer, scheduler, device, num_epochs
):
    """
    Main training loop implementing Early Stopping and Checkpointing.
    """
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_score = -np.inf
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs on device: {device}")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler, epoch
        )

        # Validate
        val_loss, val_score = validate(model, val_loader, device)

        # Update Scheduler
        if scheduler is not None:
            scheduler.step()

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val F1 Score: {val_score}")

        # Checkpointing and Early Stopping
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            # Save best model
            filename = f"{Config.MODEL_1_NAME if 'convnext' in str(type(model)).lower() else 'model'}_best.pth"
            # Adjust filename based on model name logic if needed, or just use a generic best
            # Using a generic name or passed name is better, but here we default to a standard name
            # or rely on the caller to rename. For this implementation, we use a fixed name pattern
            # but allow the utils function to handle the path.
            # We will use a generic 'model_best.pth' or specific if we can infer,
            # but to be safe and simple:
            save_checkpoint(
                model, optimizer, epoch, val_score, filename="model_best.pth"
            )
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation F1 Score: {best_score}")
