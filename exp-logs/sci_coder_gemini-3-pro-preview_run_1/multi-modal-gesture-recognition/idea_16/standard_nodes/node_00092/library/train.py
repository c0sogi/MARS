import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import (
    NUM_CLASSES,
    BACKGROUND_WEIGHT,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    NUM_EPOCHS,
    PATIENCE,
    DEVICE,
    CHECKPOINT_DIR,
    LABEL_MAP,
)
from library.utils import set_seed, decode_predictions, compute_dataset_metric
from library.data_loader import GestureDataset, collate_fn
from library.model import PCA_IIN


def get_loss_criterion(device):
    """
    Creates the CrossEntropyLoss criterion with class weights and label smoothing.
    """
    # Define class weights: Background (0) gets specific weight, others get 1.0
    weights = torch.ones(NUM_CLASSES, device=device)
    weights[0] = BACKGROUND_WEIGHT

    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=LABEL_SMOOTHING, reduction="mean"
    )
    return criterion


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for skels, audios, labels, lengths in dataloader:
        skels = skels.to(device)
        audios = audios.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Output: (Batch, Time, NumClasses)
        logits = model(skels, audios, lengths)

        # Flatten for CrossEntropyLoss
        # Logits: (Batch * Time, NumClasses)
        # Labels: (Batch * Time)
        logits_flat = logits.reshape(-1, NUM_CLASSES)
        labels_flat = labels.reshape(-1)

        loss = criterion(logits_flat, labels_flat)

        loss.backward()

        # Gradient clipping is often helpful for RNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set using Levenshtein Error Rate.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for skels, audios, labels, lengths in dataloader:
            skels = skels.to(device)
            audios = audios.to(device)
            # labels are padded (Batch, Time)

            # Forward pass
            logits = model(skels, audios, lengths)

            # Iterate over batch to decode sequences
            batch_size = logits.size(0)
            for i in range(batch_size):
                # Get valid length for this sequence
                length = int(lengths[i].item())

                # Extract valid logits for the sequence
                # (Time, NumClasses)
                seq_logits = logits[i, :length, :].cpu().numpy()

                # Decode predictions (Median Filter + RLE)
                pred_seq = decode_predictions(seq_logits)
                all_preds.append(pred_seq)

                # Extract valid target sequence
                # Remove padding and background class (0) for metric computation
                raw_target = labels[i, :length].cpu().tolist()
                target_seq = [x for x in raw_target if x != LABEL_MAP["background"]]
                all_targets.append(target_seq)

    # Compute metric
    error_rate = compute_dataset_metric(all_preds, all_targets)
    return error_rate


def train_model(debug_subset_size=None, epochs=None):
    """
    Main training function.

    Args:
        debug_subset_size (int, optional): If set, trains on a small subset of data.
        epochs (int, optional): Overrides config NUM_EPOCHS if set.
    """
    set_seed()

    # Configuration overrides
    num_epochs = epochs if epochs is not None else NUM_EPOCHS

    print(f"Initializing training on {DEVICE}...")

    # 1. Data Loaders
    train_dataset = GestureDataset(
        split="train", load_cached_data=True, debug_subset_size=debug_subset_size
    )
    val_dataset = GestureDataset(
        split="val", load_cached_data=True, debug_subset_size=debug_subset_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. Model, Criterion, Optimizer
    model = PCA_IIN().to(DEVICE)
    criterion = get_loss_criterion(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 3. Training Loop
    best_error_rate = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    print("Starting training loop...")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_error_rate = validate(model, val_loader, DEVICE)

        # Step scheduler
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Error Rate: {val_error_rate}"
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
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Error Rate: {best_error_rate}")
    return best_error_rate
