import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.utils import set_seed, decode_predictions, compute_normalized_levenshtein
from library.data_loader import GestureDataset, collate_fn
from library.model import MultiStreamGRU


def get_loss_function(device):
    """
    Creates the CrossEntropyLoss with specific class weights and label smoothing.
    Background class (0) gets weight 0.5, others get 1.0.
    """
    weights = torch.ones(Config.NUM_CLASSES, device=device)
    weights[Config.BACKGROUND_CLASS_IDX] = Config.BACKGROUND_LOSS_WEIGHT

    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=Config.LABEL_SMOOTHING
    )
    return criterion


def extract_ground_truth_sequence(labels_tensor):
    """
    Converts frame-wise label tensor to a list of gesture IDs.
    Removes background class (0) and collapses consecutive duplicates (RLE).

    Args:
        labels_tensor (torch.Tensor): 1D tensor of frame labels.

    Returns:
        list: Ordered list of gesture IDs.
    """
    labels = labels_tensor.detach().cpu().numpy()
    if len(labels) == 0:
        return []

    # 1. Collapse duplicates (RLE logic)
    collapsed = [labels[0]]
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            collapsed.append(labels[i])

    # 2. Filter out background
    sequence = [int(x) for x in collapsed if x != Config.BACKGROUND_CLASS_IDX]
    return sequence


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        if batch is None:
            continue

        # Move data to device
        skeleton = batch["skeleton"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        mask = batch["mask"].to(device)

        # Forward Pass
        # Logits: (B, T, C)
        logits = model(skeleton, audio, lengths)

        # Flatten for CrossEntropyLoss
        # We only care about valid frames defined by mask/lengths
        # Reshape: (B*T, C) and (B*T)
        logits_flat = logits.reshape(-1, Config.NUM_CLASSES)
        labels_flat = labels.reshape(-1)

        # Apply mask to select only valid frames (ignore padding)
        mask_flat = mask.reshape(-1)
        valid_logits = logits_flat[mask_flat]
        valid_labels = labels_flat[mask_flat]

        loss = criterion(valid_logits, valid_labels)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set using Levenshtein Error Rate.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            # labels = batch['labels'] # Keep on CPU for GT extraction if needed, or move and move back
            lengths = batch["lengths"].to(device)

            # Forward Pass
            logits = model(skeleton, audio, lengths)

            # Decode Predictions
            # decode_predictions handles (B, T, C) -> list of lists
            batch_preds = decode_predictions(logits)

            # Extract Ground Truths
            # batch['labels'] is (B, T)
            batch_labels = batch["labels"].numpy()
            batch_targets = []
            for i in range(batch_labels.shape[0]):
                # Slice using actual length to ignore padding in GT extraction
                length = batch["lengths"][i].item()
                seq_labels = torch.tensor(batch_labels[i, :length])
                gt_seq = extract_ground_truth_sequence(seq_labels)
                batch_targets.append(gt_seq)

            all_preds.extend(batch_preds)
            all_targets.extend(batch_targets)

    # Compute Metric
    error_rate = compute_normalized_levenshtein(all_preds, all_targets)
    return error_rate


def train_model(limit=None, epochs=Config.NUM_EPOCHS):
    """
    Main function to train the MS-DIG model.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loaders
    train_dataset = GestureDataset(split="train", limit=limit)
    val_dataset = GestureDataset(split="val", limit=limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 2. Model & Optimization
    model = MultiStreamGRU().to(device)
    criterion = get_loss_function(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 3. Training Loop
    best_error_rate = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate (Metric-Aware)
        val_error_rate = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Error Rate: {val_error_rate}"
        )

        # Checkpointing
        if val_error_rate < best_error_rate:
            best_error_rate = val_error_rate
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with Error Rate: {best_error_rate}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best Validation Error Rate: {best_error_rate}")
    return best_error_rate
