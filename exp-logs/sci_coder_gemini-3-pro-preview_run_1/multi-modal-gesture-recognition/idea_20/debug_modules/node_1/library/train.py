import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config, set_seed
from library.data_loader import GestureDataset, collate_fn
from library.model import DAGINet
from library.utils import compute_levenshtein_score, apply_median_filter, rle_decode


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        if batch is None:
            continue

        # Move data to device
        skeleton = batch["skeleton"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(skeleton, audio, lengths)

        # Flatten for loss calculation
        # Logits: (B, T, C) -> (B*T, C)
        # Labels: (B, T) -> (B*T)
        logits_flat = logits.reshape(-1, Config.NUM_CLASSES)
        labels_flat = labels.reshape(-1)

        # Compute loss
        loss = criterion(logits_flat, labels_flat)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional but recommended for RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Levenshtein Error Rate.
    """
    model.eval()
    running_loss = 0.0
    total_batches = 0

    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            # Forward pass
            logits = model(skeleton, audio, lengths)

            # Loss calculation
            logits_flat = logits.reshape(-1, Config.NUM_CLASSES)
            labels_flat = labels.reshape(-1)
            loss = criterion(logits_flat, labels_flat)
            running_loss += loss.item()
            total_batches += 1

            # Decoding for Metric
            # Get frame-wise predictions: (B, T)
            preds = torch.argmax(logits, dim=2)

            # Iterate over batch to decode sequences
            batch_size = skeleton.size(0)
            for i in range(batch_size):
                valid_len = lengths[i].item()

                # Slice valid frames
                raw_pred_seq = preds[i, :valid_len].cpu().numpy()
                raw_true_seq = labels[i, :valid_len].cpu().numpy()

                # 1. Apply Median Filter
                smoothed_pred = apply_median_filter(raw_pred_seq, kernel_size=5)

                # 2. RLE Decode
                pred_gestures = rle_decode(
                    smoothed_pred,
                    min_duration=5,
                    background_id=Config.BACKGROUND_CLASS_ID,
                )

                # Decode Ground Truth (to get the list of gesture IDs)
                true_gestures = rle_decode(
                    raw_true_seq,
                    min_duration=1,  # Ground truth might be tight
                    background_id=Config.BACKGROUND_CLASS_ID,
                )

                all_predictions.append(pred_gestures)
                all_ground_truths.append(true_gestures)

    avg_loss = running_loss / total_batches if total_batches > 0 else 0.0

    # Compute Levenshtein Score
    lev_score = compute_levenshtein_score(all_predictions, all_ground_truths)

    return avg_loss, lev_score


def run_training(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.PATIENCE,
    load_cached_data=True,
):
    """
    Main function to run the training pipeline.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Datasets & Loaders
    print("Initializing Datasets...")
    train_dataset = GestureDataset(
        split="train", load_cached_data=load_cached_data, transform=True
    )
    val_dataset = GestureDataset(
        split="val", load_cached_data=load_cached_data, transform=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 2. Model
    print("Initializing DAGI-Net...")
    model = DAGINet().to(device)

    # 3. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # Weighted CrossEntropyLoss with Label Smoothing
    class_weights = Config.get_class_weights()
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING
    )

    # 4. Training Loop
    best_lev_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_lev_score = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        duration = time.time() - start_time

        # Logging
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Time: {duration:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Levenshtein: {val_lev_score}"
        )

        # Checkpointing & Early Stopping
        if val_lev_score < best_lev_score:
            best_lev_score = val_lev_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  -> New Best Model Saved (Score: {best_lev_score})")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Levenshtein Score: {best_lev_score}")
