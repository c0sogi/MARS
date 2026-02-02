import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import (
    set_seed,
    levenshtein_distance,
    decode_predictions_to_gestures,
    median_filter_predictions,
)
from library.data_loader import get_dataloaders
from library.model import DW_AIIN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        skeleton = batch["skeleton"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)

        # Forward pass
        # Output: (B, T, NumClasses)
        logits = model(skeleton, audio, lengths)

        # Flatten for CrossEntropyLoss
        # Logits: (B*T, NumClasses)
        # Labels: (B*T)
        logits_flat = logits.reshape(-1, Config.NUM_CLASSES)
        labels_flat = labels.reshape(-1)

        # Calculate Loss
        # Padding is treated as Background class (0) and NOT masked, per instructions.
        loss = criterion(logits_flat, labels_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using Levenshtein error rate.
    """
    model.eval()

    total_distance = 0
    total_gestures = 0

    with torch.no_grad():
        for batch in loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            lengths = batch["lengths"].to(device)
            # labels are padded, we need to extract valid sequences for GT
            labels = batch["labels"].cpu().numpy()

            # Forward pass
            logits = model(skeleton, audio, lengths)

            # Get predictions (B, T)
            probs = torch.softmax(logits, dim=2)
            preds = torch.argmax(probs, dim=2).cpu().numpy()

            # Iterate over batch
            for i in range(preds.shape[0]):
                length = lengths[i].item()

                # Extract valid frames (ignore padding for decoding logic)
                # Although the model saw padding as background, for metric we look at the valid duration
                # defined by the original sequence length.
                valid_pred = preds[i, :length]
                valid_label = labels[i, :length]

                # 1. Apply Median Filter
                smoothed_pred = median_filter_predictions(
                    valid_pred, window_size=Config.MEDIAN_FILTER_WINDOW
                )

                # 2. Decode to Gesture List (RLE + Filtering)
                pred_seq = decode_predictions_to_gestures(
                    smoothed_pred,
                    background_label=Config.BACKGROUND_LABEL,
                    min_length=Config.MIN_GESTURE_LENGTH,
                )

                true_seq = decode_predictions_to_gestures(
                    valid_label,
                    background_label=Config.BACKGROUND_LABEL,
                    min_length=1,  # GT usually doesn't need min_length filtering, but we filter background
                )

                # 3. Compute Levenshtein Distance
                dist = levenshtein_distance(pred_seq, true_seq)

                total_distance += dist
                total_gestures += len(true_seq)

    # Error Rate = Total Distance / Total GT Gestures
    # Can exceed 1.0
    if total_gestures == 0:
        return 0.0

    error_rate = total_distance / total_gestures
    return error_rate


def run_training(num_epochs=Config.NUM_EPOCHS, load_cached_data=True):
    """
    Main training routine.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loaders
    train_loader, val_loader, _ = get_dataloaders()

    # 2. Model
    model = DW_AIIN().to(device)

    # 3. Loss Function
    # Weights: Background=0.5, Others=1.0
    class_weights = Config.get_loss_weights()
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING
    )

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 5. Training Loop
    best_error_rate = float("inf")
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_error_rate = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        duration = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Time: {duration:.2f}s | "
            f"Train Loss: {train_loss:.10f} | "
            f"Val Error Rate: {val_error_rate:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_error_rate < best_error_rate:
            best_error_rate = val_error_rate
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved to {Config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Validation Error Rate: {best_error_rate:.10f}")
