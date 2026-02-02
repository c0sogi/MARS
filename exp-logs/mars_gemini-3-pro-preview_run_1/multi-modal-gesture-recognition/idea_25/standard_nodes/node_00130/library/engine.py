import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.model import MPWINet
from library.utils import median_filter, rle_decode, levenshtein_distance


def get_loss_function(device):
    """
    Creates the CrossEntropyLoss with specific class weights and label smoothing.
    Background class (index 0) gets weight 0.5.
    """
    # Weights: 0.5 for Background, 1.0 for all others
    weights = torch.ones(Config.NUM_CLASSES, device=device)
    weights[Config.BACKGROUND_LABEL] = Config.BACKGROUND_WEIGHT

    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=Config.LABEL_SMOOTHING, reduction="mean"
    )
    return criterion


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move inputs to device
        skeleton = batch["skeleton"].to(device)
        audio = batch["audio"].to(device)
        lengths = batch[
            "lengths"
        ]  # Keep on CPU for pack_padded_sequence if needed, or move inside model
        mask = batch["mask"].to(device)
        frame_labels = batch["frame_labels"].to(device)

        # Forward pass
        # Model expects: skeleton, audio, lengths, mask
        logits = model(skeleton, audio, lengths, mask)  # (B, T, NumClasses)

        # Reshape for Loss: (B*T, NumClasses) vs (B*T)
        # We do not mask padding for loss calculation as per instructions (Supervised Padding)
        logits_flat = logits.view(-1, Config.NUM_CLASSES)
        targets_flat = frame_labels.view(-1)

        loss = criterion(logits_flat, targets_flat)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional but good for RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Levenshtein Error Rate.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    total_dist = 0
    total_truth_len = 0

    with torch.no_grad():
        for batch in dataloader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            lengths = batch["lengths"]
            mask = batch["mask"].to(device)
            frame_labels = batch["frame_labels"].to(device)

            # Ground truth sequences (list of tensors/lists)
            labels_seq_gt = batch["labels_seq"]

            # Forward
            logits = model(skeleton, audio, lengths, mask)  # (B, T, C)

            # Loss
            logits_flat = logits.view(-1, Config.NUM_CLASSES)
            targets_flat = frame_labels.view(-1)
            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item()
            num_batches += 1

            # Decoding & Metric Calculation
            # Get hard predictions
            preds = torch.argmax(logits, dim=2)  # (B, T)
            preds_np = preds.cpu().numpy()
            lengths_np = lengths.numpy()

            for i in range(preds.shape[0]):
                # Slice valid frames
                valid_len = lengths_np[i]
                raw_pred_seq = preds_np[i, :valid_len]

                # Post-processing
                smoothed_seq = median_filter(
                    raw_pred_seq, window_size=Config.MEDIAN_FILTER_WINDOW
                )
                decoded_gestures = rle_decode(
                    smoothed_seq,
                    background_label=Config.BACKGROUND_LABEL,
                    min_len=Config.MIN_SEGMENT_LENGTH,
                )

                # Get Ground Truth
                # labels_seq_gt[i] might be a tensor or list
                gt_seq = labels_seq_gt[i]
                if isinstance(gt_seq, torch.Tensor):
                    gt_seq = gt_seq.tolist()

                # Compute Distance
                dist = levenshtein_distance(decoded_gestures, gt_seq)

                total_dist += dist
                total_truth_len += len(gt_seq)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Metric: Error Rate = Total Distance / Total Truth Gestures
    # Can exceed 1.0
    metric = total_dist / total_truth_len if total_truth_len > 0 else 0.0

    return avg_loss, metric


def train_model(train_loader, val_loader):
    """
    Main training routine.
    """
    device = Config.DEVICE
    print(f"Training on device: {device}")

    # Initialize Model
    model = MPWINet().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # Loss Function
    criterion = get_loss_function(device)

    # Early Stopping Tracking
    best_metric = float("inf")
    patience = 10
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric (Levenshtein): {val_metric}"
        )

        # Checkpoint & Early Stopping
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved! Metric: {best_metric}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation Metric: {best_metric}")
    return best_model_path
