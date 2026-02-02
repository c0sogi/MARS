import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import scipy.signal
import time

from library.config import (
    CHECKPOINT_DIR,
    SEED,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    MEDIAN_FILTER_KERNEL,
    NUM_CLASSES,
    VAL_METADATA_PATH,
    TRAIN_METADATA_PATH,
)
from library.utils import set_seed, compute_levenshtein_score, load_metadata
from library.dataset import GestureDataset, collate_fn
from library.model import SBG_CRCN
from library.loss import ActionSegmentationLoss


def get_ground_truth_map(split="val"):
    """
    Loads the metadata and creates a dictionary mapping sample_id to the list of ground truth labels.
    """
    df = load_metadata(split)
    return pd.Series(df.labels.values, index=df.sample_id).to_dict()


def decode_predictions(class_probs, mask):
    """
    Decodes frame-level probabilities into a sequence of gesture labels.
    Applies Median Filtering, collapses repeats, and removes background.

    Args:
        class_probs (torch.Tensor): (B, T, C) Softmax probabilities.
        mask (torch.Tensor): (B, T) Boolean mask.

    Returns:
        list of lists: Predicted gesture sequences for the batch.
    """
    predictions = []

    # Convert to numpy for processing
    class_probs_np = class_probs.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    B, T, C = class_probs_np.shape

    for b in range(B):
        # Get valid length
        valid_len = int(np.sum(mask_np[b]))
        if valid_len == 0:
            predictions.append([])
            continue

        # Get frame labels: Argmax
        probs_b = class_probs_np[b, :valid_len, :]
        frame_labels = np.argmax(probs_b, axis=1)

        # Apply Median Filter to smooth
        if MEDIAN_FILTER_KERNEL > 1:
            # Kernel size must be odd
            k = (
                MEDIAN_FILTER_KERNEL
                if MEDIAN_FILTER_KERNEL % 2 == 1
                else MEDIAN_FILTER_KERNEL + 1
            )
            frame_labels = scipy.signal.medfilt(frame_labels, kernel_size=k)

        # Collapse repeats and remove background (0)
        sequence = []
        last_label = -1

        for label in frame_labels:
            if label != last_label:
                if label != 0:  # 0 is background
                    sequence.append(int(label))
                last_label = label

        predictions.append(sequence)

    return predictions


def validate(model, dataloader, criterion, gt_map, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            class_targets = batch["class_target"].to(device)
            boundary_targets = batch["boundary_target"].to(device)
            mask = batch["mask"].to(device)
            sample_ids = batch["sample_ids"]

            # Forward
            outputs = model(features, mask)

            # Compute Loss
            loss, _ = criterion(outputs, class_targets, boundary_targets, mask)
            total_loss += loss.item()

            # Decode Predictions (using Stage 3 output)
            stage3_probs = outputs["stage3"]["class_probs"]
            batch_preds = decode_predictions(stage3_probs, mask)

            # Get Ground Truths
            batch_gts = [gt_map.get(sid, []) for sid in sample_ids]

            all_preds.extend(batch_preds)
            all_gts.extend(batch_gts)

    avg_loss = total_loss / len(dataloader)
    levenshtein_score = compute_levenshtein_score(all_preds, all_gts)

    return avg_loss, levenshtein_score


def train_model(
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    patience=PATIENCE,
    load_cached_data=True,
    augment=True,
):
    """
    Main training loop.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Data Loading
    print("Initializing Datasets...")
    train_dataset = GestureDataset(
        split="train", augment=augment, load_cached_data=load_cached_data
    )
    val_dataset = GestureDataset(
        split="val", augment=False, load_cached_data=load_cached_data
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Load GT maps for validation metric
    val_gt_map = get_ground_truth_map("val")

    # 2. Model & Optimization
    print("Initializing Model...")
    model = SBG_CRCN().to(device)
    criterion = ActionSegmentationLoss().to(device)

    # Separate parameter groups if needed, but standard AdamW usually works fine
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )

    # Scheduler: Reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2, verbose=True
    )

    # 3. Training Loop
    best_lev_score = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    print("Starting Training...")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        train_loss_accum = 0.0

        # Training Step
        for batch in train_loader:
            features = batch["features"].to(device)
            class_targets = batch["class_target"].to(device)
            boundary_targets = batch["boundary_target"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            outputs = model(features, mask)
            loss, _ = criterion(outputs, class_targets, boundary_targets, mask)

            loss.backward()

            # Gradient clipping to prevent exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation Step
        val_loss, val_lev_score = validate(
            model, val_loader, criterion, val_gt_map, device
        )

        # Scheduler Step
        scheduler.step(val_lev_score)

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch}/{num_epochs} | Time: {epoch_time:.1f}s | "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Val Levenshtein: {val_lev_score:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_lev_score < best_lev_score:
            best_lev_score = val_lev_score
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model Saved (Score: {best_lev_score:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  >>> No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print("Early Stopping Triggered.")
            break

    print(f"Training Complete. Best Validation Score: {best_lev_score:.6f}")
    return best_model_path
