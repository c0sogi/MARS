import os
import time
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from scipy.signal import medfilt

from library import config, utils, data_loader, model, losses


def decode_sequence(frame_labels, valid_length, filter_size=7):
    """
    Decodes frame-wise labels into a sequence of gesture IDs.
    Applies Median Filtering, collapses repeats, and removes background (0).
    """
    # Truncate to valid length
    labels = frame_labels[:valid_length]

    if len(labels) == 0:
        return []

    # 1. Median Filtering (Smoothing)
    # Ensure filter_size is odd
    if filter_size % 2 == 0:
        filter_size += 1

    # medfilt requires 1D array
    if len(labels) >= filter_size:
        smoothed_labels = medfilt(labels, kernel_size=filter_size).astype(int)
    else:
        smoothed_labels = labels

    # 2. Collapse Repeats and Remove Background
    sequence = []
    last_label = -1

    for lbl in smoothed_labels:
        if lbl != last_label:
            if lbl != 0:  # 0 is background
                sequence.append(int(lbl))
            last_label = lbl

    return sequence


def train_one_epoch(model_instance, loader, optimizer, loss_fn, device):
    """
    Performs one epoch of training.
    """
    model_instance.train()
    total_loss = 0.0
    metrics_accum = {}

    for batch in loader:
        # Move inputs to device
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)

        # Move targets to device
        targets = {
            "labels": batch["labels"].to(device),
            "boundaries": batch["boundaries"].to(device),
            "mask": mask,
        }

        # Forward pass
        optimizer.zero_grad()
        outputs = model_instance(features, mask, lengths)

        # Compute loss
        loss, batch_metrics = loss_fn(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional but recommended for LSTMs)
        torch.nn.utils.clip_grad_norm_(model_instance.parameters(), max_norm=5.0)

        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        for k, v in batch_metrics.items():
            metrics_accum[k] = metrics_accum.get(k, 0.0) + v

    # Average metrics
    avg_loss = total_loss / len(loader)
    avg_metrics = {k: v / len(loader) for k, v in metrics_accum.items()}

    return avg_loss, avg_metrics


def validate(model_instance, loader, loss_fn, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Levenshtein Error Rate.
    """
    model_instance.eval()
    total_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)

            targets_gpu = {
                "labels": batch["labels"].to(device),
                "boundaries": batch["boundaries"].to(device),
                "mask": mask,
            }

            # Forward pass
            outputs = model_instance(features, mask, lengths)

            # Compute loss
            loss, _ = loss_fn(outputs, targets_gpu)
            total_loss += loss.item()

            # --- Decode for Metric Calculation ---
            # Use Stage 3 output for final predictions
            # outputs['stage3'] is (cls_logits, bnd_logits)
            # cls_logits is (B, T, C)
            stage3_logits = outputs["stage3"][0]
            stage3_preds = torch.argmax(stage3_logits, dim=2).cpu().numpy()  # (B, T)

            # Get Ground Truth from batch (cpu)
            batch_labels = batch["labels"].numpy()  # (B, T)
            batch_lengths = batch["lengths"].numpy()

            for i in range(len(batch_lengths)):
                valid_len = batch_lengths[i]

                # Decode Prediction
                pred_seq = decode_sequence(stage3_preds[i], valid_len)
                all_preds.append(pred_seq)

                # Decode Target (Reconstruct sequence from frame-wise labels)
                # We use the same logic to ensure consistency (collapse repeats, remove background)
                target_seq = decode_sequence(
                    batch_labels[i], valid_len, filter_size=1
                )  # No smoothing for GT
                all_targets.append(target_seq)

    avg_loss = total_loss / len(loader)

    # Compute Levenshtein Error Rate
    error_rate = utils.compute_levenshtein(all_preds, all_targets)

    return avg_loss, error_rate


def run_training(
    num_epochs=config.NUM_EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    load_cached_data=True,
):
    """
    Main training loop.
    """
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Datasets
    print("Initializing datasets...")
    train_dataset = data_loader.GestureDataset(
        config.TRAIN_METADATA_PATH, is_train=True, load_cached_data=load_cached_data
    )
    val_dataset = data_loader.GestureDataset(
        config.VAL_METADATA_PATH, is_train=False, load_cached_data=load_cached_data
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=data_loader.collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=data_loader.collate_fn,
        pin_memory=True,
    )

    # 2. Initialize Model, Loss, Optimizer
    print("Initializing model...")
    net = model.CASGCN().to(device)

    criterion = losses.MultiStageLoss(device=device)

    optimizer = optim.AdamW(
        net.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler (Optional but good for convergence)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    save_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_metrics = train_one_epoch(
            net, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_error_rate = validate(net, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        # Logging
        print(f"Epoch {epoch}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.8f}")
        print(f"  Val Loss:   {val_loss:.8f}")
        print(f"  Val Error:  {val_error_rate:.8f}")

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(net.state_dict(), save_path)
            print(f"  New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
