import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import (
    NUM_CLASSES,
    BACKGROUND_CLASS_ID,
    BACKGROUND_WEIGHT,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    GRADIENT_CLIP_VAL,
    BEST_MODEL_PATH,
    SEED,
)
from library.utils import set_seed, levenshtein_distance, rle_decode, median_filter
from library.data_loader import get_dataloaders
from library.model import DGR_RN


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        skeleton = batch["skeleton"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)  # (B, T)

        optimizer.zero_grad()

        # Forward pass
        logits = model(skeleton, audio)  # (B, T, C)

        # Flatten and mask for loss calculation
        # We only want to compute loss on valid frames, not padding
        # Reshape to (B*T, C) and (B*T)
        logits_flat = logits.reshape(-1, NUM_CLASSES)
        labels_flat = labels.reshape(-1)
        mask_flat = mask.reshape(-1)

        # Filter out padded elements
        valid_logits = logits_flat[mask_flat]
        valid_labels = labels_flat[mask_flat]

        if valid_labels.numel() > 0:
            loss = criterion(valid_logits, valid_labels)

            # Backward
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, loader, device):
    model.eval()

    total_dist = 0
    total_ref_gestures = 0

    with torch.no_grad():
        for batch in loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)  # (B, T)
            lengths = batch["lengths"]

            # Forward
            logits = model(skeleton, audio)  # (B, T, C)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)  # (B, T)

            # Convert to CPU for decoding
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            probs_np = probs.cpu().numpy()

            # Iterate over batch
            for i in range(len(labels)):
                length = lengths[i]

                # Extract valid sequence based on length
                # Note: Median filter expects (Time, Channels) or (Time,)
                curr_probs = probs_np[i, :length, :]
                curr_labels = labels_np[i, :length]

                # Apply Median Filter to probabilities before argmax for smoother predictions
                # Or apply to the argmax output.
                # The prompt strategy says: "Apply a Median Filter (window size 5) to smooth the frame-wise probability maps"
                smoothed_probs = median_filter(curr_probs, window_size=5)
                smoothed_preds = np.argmax(smoothed_probs, axis=-1)

                # Decode
                hyp_seq = rle_decode(smoothed_preds)

                # For Ground Truth, we also use rle_decode to get the sequence of IDs from dense labels
                # This handles the background class filtering automatically
                ref_seq = rle_decode(curr_labels)

                # Compute Metric
                dist = levenshtein_distance(hyp_seq, ref_seq)

                total_dist += dist
                total_ref_gestures += len(ref_seq)

    ler = total_dist / total_ref_gestures if total_ref_gestures > 0 else 0.0
    return ler


def run_training(stats_path=None):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(stats_path=stats_path)

    # Model
    model = DGR_RN().to(device)

    # Loss
    # Class weights: 0.5 for background (index 0), 1.0 for others
    weights = torch.ones(NUM_CLASSES).to(device)
    weights[BACKGROUND_CLASS_ID] = BACKGROUND_WEIGHT

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training Loop
    best_ler = float("inf")
    patience = 10
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_ler = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val LER: {val_ler}"
        )

        # Checkpointing
        if val_ler < best_ler:
            best_ler = val_ler
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New best model saved with LER: {best_ler}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best Validation LER: {best_ler}")
