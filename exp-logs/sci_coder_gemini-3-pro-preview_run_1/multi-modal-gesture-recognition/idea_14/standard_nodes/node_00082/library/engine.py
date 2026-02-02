import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    BACKGROUND_LABEL,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    SEED,
)
from library.data_utils import compute_global_stats
from library.dataset import GestureDataset, collate_fn
from library.model import GCAResNet
from library.metrics import evaluate_batch, decode_predictions

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def get_dataloaders(batch_size=BATCH_SIZE):
    """
    Initializes datasets and dataloaders for training and validation.
    Computes/loads global stats for normalization.
    """
    # Load Metadata
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    # Compute Global Stats (Cached)
    # We use the training set to compute stats
    stats = compute_global_stats(train_df, load_cached_data=True)

    # Initialize Datasets
    train_ds = GestureDataset(train_df, stats=stats, is_train=True, augment=True)
    val_ds = GestureDataset(val_df, stats=stats, is_train=False, augment=False)

    # Initialize Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, stats


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        if batch is None:
            continue

        # Move data to device
        skeletons = batch["skeleton"].to(device)
        audios = batch["audio"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        # Forward Pass
        # logits: (B, T, C)
        logits = model(skeletons, audios, lengths)

        # Prepare for CTC Loss
        # CTC expects LogSoftmax inputs of shape (T, B, C)
        log_probs = nn.functional.log_softmax(logits, dim=2).permute(1, 0, 2)

        # Calculate Loss
        # labels are padded with BACKGROUND_LABEL (0), which acts as blank for CTC
        # We must rely on label_lengths to ignore padding
        loss = criterion(log_probs, labels, lengths, label_lengths)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def validate(model, dataloader, criterion, device):
    """
    Runs validation and computes Levenshtein Error Rate.
    """
    model.eval()
    total_loss = 0.0
    total_dist = 0
    total_len = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            skeletons = batch["skeleton"].to(device)
            audios = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            # Forward
            logits = model(skeletons, audios, lengths)

            # Loss
            log_probs = nn.functional.log_softmax(logits, dim=2).permute(1, 0, 2)
            loss = criterion(log_probs, labels, lengths, label_lengths)
            total_loss += loss.item()

            # Metrics (Levenshtein Distance)
            # evaluate_batch handles decoding and comparison
            batch_dist, batch_len = evaluate_batch(logits, lengths, labels)
            total_dist += batch_dist
            total_len += batch_len
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    # Error Rate = Total Distance / Total Ground Truth Length
    error_rate = total_dist / max(1, total_len)

    return avg_loss, error_rate


def fit(num_epochs=NUM_EPOCHS):
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    # Setup
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_loader, val_loader, stats = get_dataloaders()

    model = GCAResNet().to(DEVICE)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Loss Function
    # Using CTCLoss as we have sequence labels without frame alignment.
    # blank=BACKGROUND_LABEL (0) matches our config.
    criterion = nn.CTCLoss(blank=BACKGROUND_LABEL, reduction="mean", zero_infinity=True)

    best_error_rate = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {DEVICE} for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch
        )

        # Validate
        val_loss, val_error_rate = validate(model, val_loader, criterion, DEVICE)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Error Rate: {val_error_rate:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_error_rate < best_error_rate:
            best_error_rate = val_error_rate
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Error Rate: {best_error_rate:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    print(f"Training complete. Best Validation Error Rate: {best_error_rate:.6f}")
    return best_model_path, stats


def generate_submission(model_path, stats):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating submission...")

    # Load Test Data
    if not os.path.exists(TEST_CSV):
        print("Test metadata not found.")
        return

    test_df = pd.read_csv(TEST_CSV)
    test_ds = GestureDataset(test_df, stats=stats, is_train=False, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Load Model
    model = GCAResNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            skeletons = batch["skeleton"].to(DEVICE)
            audios = batch["audio"].to(DEVICE)
            lengths = batch["lengths"].to(DEVICE)
            sample_ids = batch["sample_ids"]

            # Inference
            logits = model(skeletons, audios, lengths)

            # Decode
            # decode_predictions returns List[List[int]]
            batch_preds = decode_predictions(logits, lengths)

            for sid, preds in zip(sample_ids, batch_preds):
                # Format: SessionID,1,2,3
                pred_str = ",".join(map(str, preds))
                results.append(f"{sid},{pred_str}")

    # Save to CSV
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    with open(submission_path, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Submission saved to {submission_path}")


def run_pipeline():
    """
    Runs the full training and submission pipeline.
    """
    best_model_path, stats = fit()
    generate_submission(best_model_path, stats)
