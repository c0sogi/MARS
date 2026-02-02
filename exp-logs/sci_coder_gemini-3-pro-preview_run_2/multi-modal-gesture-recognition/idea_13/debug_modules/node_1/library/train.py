import os
import torch
import numpy as np
import scipy.ndimage
from library.config import Config
from library.utils import set_seed, get_device, compute_score
from library.model import SBMD_CRCN
from library.losses import CombinedLoss
from library.data_loader import get_dataloaders


def decode_sequence(frame_labels):
    """
    Decodes a frame-wise label sequence into a list of gesture IDs.
    Applies:
    1. Median Filtering (Smoothing)
    2. Collapse consecutive duplicates
    3. Remove background (0)
    """
    # 1. Median Filter
    # mode='nearest' handles boundary padding
    smoothed_labels = scipy.ndimage.median_filter(
        frame_labels, size=Config.MEDIAN_FILTER_KERNEL, mode="nearest"
    )

    # 2. Collapse duplicates
    unique_sequence = []
    if len(smoothed_labels) > 0:
        unique_sequence.append(smoothed_labels[0])
        for i in range(1, len(smoothed_labels)):
            if smoothed_labels[i] != smoothed_labels[i - 1]:
                unique_sequence.append(smoothed_labels[i])

    # 3. Remove background (0)
    gesture_sequence = [x for x in unique_sequence if x != 0]

    return gesture_sequence


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        cls_targets = batch["cls_targets"].to(device)
        bnd_targets = batch["bnd_targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, mask)

        # Compute loss
        # Prepare targets dict for CombinedLoss
        targets = {"cls_targets": cls_targets, "bnd_targets": bnd_targets, "mask": mask}

        loss_dict = criterion(outputs, targets)
        loss = loss_dict["loss"]

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_truths = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            cls_targets = batch["cls_targets"].to(device)
            bnd_targets = batch["bnd_targets"].to(device)

            # Forward pass
            outputs = model(features, mask)

            # Compute loss
            targets = {
                "cls_targets": cls_targets,
                "bnd_targets": bnd_targets,
                "mask": mask,
            }
            loss_dict = criterion(outputs, targets)
            total_loss += loss_dict["loss"].item()
            num_batches += 1

            # Decoding for metric calculation
            # Use Stage 3 outputs
            s3_logits = outputs["stage3_cls"]  # (B, T, C)
            s3_probs = torch.softmax(s3_logits, dim=-1)
            s3_preds = torch.argmax(s3_probs, dim=-1)  # (B, T)

            # Convert to CPU numpy
            preds_np = s3_preds.cpu().numpy()
            targets_np = cls_targets.cpu().numpy()
            masks_np = mask.cpu().numpy()

            # Process each sample in the batch
            for i in range(preds_np.shape[0]):
                # Get valid length based on mask
                valid_len = int(masks_np[i].sum())

                # Slice valid frames
                pred_seq_raw = preds_np[i, :valid_len]
                target_seq_raw = targets_np[i, :valid_len]

                # Decode
                pred_gestures = decode_sequence(pred_seq_raw)
                true_gestures = decode_sequence(target_seq_raw)

                all_preds.append(pred_gestures)
                all_truths.append(true_gestures)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Compute Levenshtein Score
    score = compute_score(all_preds, all_truths)

    return avg_loss, score


def run_training():
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing model...")
    model = SBMD_CRCN().to(device)

    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = CombinedLoss().to(device)

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_score = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Score (Levenshtein Error): {val_score}"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with score: {best_score}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")
