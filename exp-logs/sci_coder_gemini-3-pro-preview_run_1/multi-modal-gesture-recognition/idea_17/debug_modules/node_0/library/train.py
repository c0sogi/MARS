import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import (
    DEVICE,
    CHECKPOINT_DIR,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    PATIENCE,
    BATCH_SIZE,
    NUM_WORKERS,
    BACKGROUND_CLASS_ID,
    MODEL_OUTPUT_CLASSES,
    BACKGROUND_WEIGHT,
    LABEL_SMOOTHING,
    SEED,
)
from library.model import GCINet
from library.data_loader import get_loaders
from library.utils import set_seed, levenshtein_distance, decode_predictions


def get_ground_truth_sequence(labels_tensor):
    """
    Converts a frame-wise label tensor into a sequence of gesture IDs.
    Applies simple RLE and filters out background class.

    Args:
        labels_tensor (torch.Tensor): Shape (T,)

    Returns:
        list: Ordered list of gesture IDs.
    """
    labels = labels_tensor.cpu().numpy()
    if len(labels) == 0:
        return []

    # Run-Length Encoding
    segments = []
    if len(labels) > 0:
        current_label = labels[0]
        # We don't strictly need length for GT extraction, just the change points,
        # but following a similar logic to prediction decoding is good practice.

        for label in labels[1:]:
            if label != current_label:
                segments.append(current_label)
                current_label = label
        segments.append(current_label)

    # Filter background
    final_sequence = [int(x) for x in segments if x != BACKGROUND_CLASS_ID]
    return final_sequence


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Unpack batch
        skeleton = batch["skeleton"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(skeleton, audio, lengths)  # (B, T, Classes)

        # Reshape for Loss: (B*T, Classes) vs (B*T)
        # We do NOT mask padding; padding is treated as Background class (0)
        logits_flat = logits.view(-1, MODEL_OUTPUT_CLASSES)
        labels_flat = labels.view(-1)

        loss = criterion(logits_flat, labels_flat)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def validate(model, loader, device):
    model.eval()
    total_distance = 0
    total_gt_gestures = 0

    with torch.no_grad():
        for batch in loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            logits = model(skeleton, audio, lengths)

            # Iterate over batch to decode sequences
            for i in range(logits.size(0)):
                length = lengths[i].item()

                # Get valid portion of logits and labels
                # Note: decode_predictions expects logits, we pass the valid time steps
                valid_logits = logits[i, :length, :]
                valid_labels = labels[i, :length]

                # Decode Prediction
                pred_seq = decode_predictions(valid_logits)

                # Decode Ground Truth
                gt_seq = get_ground_truth_sequence(valid_labels)

                # Compute Metric
                dist = levenshtein_distance(pred_seq, gt_seq)

                total_distance += dist
                total_gt_gestures += len(gt_seq)

    # Avoid division by zero
    if total_gt_gestures == 0:
        return 0.0

    error_rate = total_distance / total_gt_gestures
    return error_rate


def train_model(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, debug_subset_size=None):
    set_seed(SEED)

    # 1. Data Loaders
    # Note: debug_subset_size is handled inside data_loader via config,
    # but if we wanted to override dynamically we would need to modify config or loader.
    # Assuming config.DEBUG_SUBSET_SIZE is set appropriately or we rely on defaults.
    train_loader, val_loader, _ = get_loaders(
        batch_size=batch_size, num_workers=NUM_WORKERS
    )

    # 2. Model Setup
    model = GCINet().to(DEVICE)

    # 3. Loss Setup
    # Background Class Weight (0.7), others 1.0
    class_weights = torch.ones(MODEL_OUTPUT_CLASSES, device=DEVICE)
    class_weights[BACKGROUND_CLASS_ID] = BACKGROUND_WEIGHT

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
        reduction="mean",
        # Note: 'mean' averages over all tokens (B*T) including padding (which are background).
        # This anchors null-state predictions as requested.
    )

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 5. Training Loop
    best_error_rate = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {DEVICE} for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_error_rate = validate(model, val_loader, DEVICE)

        # Step Scheduler
        scheduler.step()

        # Logging (Full precision)
        print(
            f"Epoch {epoch}: Train Loss = {train_loss}, Val Error Rate = {val_error_rate}"
        )

        # Checkpointing & Early Stopping
        if val_error_rate < best_error_rate:
            best_error_rate = val_error_rate
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Error Rate: {best_error_rate}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best Validation Error Rate: {best_error_rate}")
    return best_model_path
