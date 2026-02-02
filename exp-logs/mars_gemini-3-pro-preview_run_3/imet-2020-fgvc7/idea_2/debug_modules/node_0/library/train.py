import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.dataset import get_dataloaders
from library.model import ArtworkModel
from library.utils import set_seed, calculate_f1, optimize_threshold


def apply_label_smoothing(targets, smoothing=0.0):
    """
    Applies label smoothing to binary targets.
    Formula: new_y = y * (1 - alpha) + 0.5 * alpha
    This maps 0 -> 0.5*alpha and 1 -> 1 - 0.5*alpha
    """
    if smoothing <= 0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device, epoch
):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # Apply label smoothing to targets
        smoothed_targets = apply_label_smoothing(targets, Config.LABEL_SMOOTHING)

        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss = criterion(logits, smoothed_targets)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Validates the model and returns loss, F1 score, and predictions.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            with autocast():
                logits = model(images)
                # Calculate loss against raw targets or smoothed?
                # Usually validation loss is calculated against ground truth,
                # but for consistency with train we can use raw targets (smoothing=0).
                loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # Calculate F1 with default threshold for monitoring progress
    y_pred_bin = (all_probs > Config.DEFAULT_THRESHOLD).astype(int)
    val_f1 = calculate_f1(all_targets, y_pred_bin)

    return epoch_loss, val_f1, all_probs, all_targets


def run_training(debug=Config.DEBUG):
    """
    Main training pipeline.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # Initialize Model
    model = ArtworkModel(pretrained=True)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    scaler = GradScaler()

    best_f1 = -1.0
    best_epoch = 0

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )
        val_loss, val_f1, val_probs, val_targets = validate(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - start_time

        # Print full precision as requested
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.2f}s | Train Loss: {train_loss} | Val Loss: {val_loss} | Val F1: {val_f1}"
        )

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved at epoch {epoch}")

    print(f"Training complete. Best F1: {best_f1} at epoch {best_epoch}")

    # Load best model for threshold optimization
    print("Loading best model for threshold optimization...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Get predictions on validation set using best model
    _, _, val_probs, val_targets = validate(model, val_loader, criterion, device)

    # Optimize threshold
    best_threshold, best_opt_f1 = optimize_threshold(val_targets, val_probs)

    return model, best_threshold


def generate_submission(model, threshold, device, debug=Config.DEBUG):
    """
    Generates submission file using TTA and optimized threshold.
    """
    print("Generating submission...")
    _, _, test_loader = get_dataloaders(debug=debug)

    model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for images, batch_ids in test_loader:
            images = images.to(device)

            # TTA: Original
            with autocast():
                logits_orig = model(images)
                probs_orig = torch.sigmoid(logits_orig)

            # TTA: Horizontal Flip
            images_flipped = torch.flip(images, dims=[3])
            with autocast():
                logits_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped)

            # Average probabilities
            avg_probs = (probs_orig + probs_flipped) / 2.0

            # Apply threshold
            preds_bin = (avg_probs > threshold).cpu().numpy().astype(int)

            # Format predictions
            for i, pred_row in enumerate(preds_bin):
                # Get indices of active classes
                indices = np.where(pred_row == 1)[0]
                pred_str = " ".join(map(str, indices))
                predictions.append(pred_str)
                ids.append(batch_ids[i])

    # Create submission DataFrame
    sub_df = pd.DataFrame({"id": ids, "attribute_ids": predictions})

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    """
    Main entry point for training and submission.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Train and get best model/threshold
    model, best_threshold = run_training()

    # Generate submission
    generate_submission(model, best_threshold, Config.DEVICE)
