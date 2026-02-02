import torch
import torch.nn as nn
import numpy as np
import os
from library.config import Config
from library.utils import f05_score


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    scaler = torch.amp.GradScaler("cuda")

    # Zero the parameter gradients
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # Forward pass
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, masks)
            # Normalize loss for gradient accumulation
            loss = loss / Config.GRAD_ACCUM_STEPS

        # Backward pass
        scaler.scale(loss).backward()

        # Optimize every GRAD_ACCUM_STEPS or at the end of the epoch
        if (i + 1) % Config.GRAD_ACCUM_STEPS == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * Config.GRAD_ACCUM_STEPS
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Calculates loss and searches for the best F0.5 score threshold.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()
            num_batches += 1

            # Apply sigmoid to convert logits to probabilities for metric calculation
            probs = torch.sigmoid(outputs)

            # Store on CPU to save GPU memory for accumulation
            all_preds.append(probs.cpu())
            all_targets.append(masks.cpu())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate all batches
    if all_preds:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
    else:
        return avg_loss, 0.0, 0.5

    # Threshold Search for best F0.5
    best_score = -1.0
    best_threshold = 0.5

    # Generate range of thresholds from Config
    thresholds = np.arange(
        Config.THRESHOLD_SEARCH_START,
        Config.THRESHOLD_SEARCH_END + 1e-6,
        Config.THRESHOLD_SEARCH_STEP,
    )

    for thresh in thresholds:
        score = f05_score(all_preds, all_targets, threshold=thresh)
        if score > best_score:
            best_score = score
            best_threshold = thresh

    return avg_loss, best_score, best_threshold


def run_training(model, train_loader, val_loader, optimizer, device):
    """
    Orchestrates the training loop with early stopping.
    """
    # Define Loss Function with positive weight handling
    # pos_weight must be a tensor on the correct device
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_score = -1.0
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    print(f"Device: {device}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_score, val_thresh = evaluate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(f"Epoch {epoch}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val F0.5 Score: {val_score} (at threshold {val_thresh})")

        # Early Stopping and Checkpointing
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            print(f"New best model saved to {Config.CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    return best_val_score
