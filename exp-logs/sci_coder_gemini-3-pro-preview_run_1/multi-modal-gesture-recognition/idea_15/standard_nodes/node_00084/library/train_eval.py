import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import itertools
from library.config import Config
from library.utils import set_seed, compute_error_rate, rle_decode
from library.data_loader import get_dataloaders
from library.model import GCA_IIN


def decode_target_seq(target_tensor):
    """
    Decodes a dense frame-wise target tensor into a list of gesture IDs.
    Groups consecutive identical values and removes background (0).
    """
    # target_tensor: 1D numpy array or torch tensor
    if isinstance(target_tensor, torch.Tensor):
        target_tensor = target_tensor.cpu().numpy()

    # Group consecutive values
    grouped = [k for k, g in itertools.groupby(target_tensor)]
    # Filter out background (0)
    return [int(k) for k in grouped if k != Config.BACKGROUND_CLASS_ID]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    count = 0

    for batch in loader:
        skeleton = batch["skeleton"].to(device)
        audio = batch["audio"].to(device)
        targets = batch["target"].to(device)
        lengths = batch["lengths"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # logits: (B, T, C)
        logits = model(skeleton, audio, lengths)

        # Flatten for CrossEntropy: (B*T, C) vs (B*T)
        logits_flat = logits.reshape(-1, Config.NUM_CLASSES)
        targets_flat = targets.reshape(-1)

        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        count += targets.size(0)

    return total_loss / count if count > 0 else 0.0


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            targets = batch["target"].to(device)
            lengths = batch["lengths"].to(device)

            logits = model(skeleton, audio, lengths)

            # Loss calculation
            logits_flat = logits.reshape(-1, Config.NUM_CLASSES)
            targets_flat = targets.reshape(-1)
            loss = criterion(logits_flat, targets_flat)

            total_loss += loss.item() * targets.size(0)
            count += targets.size(0)

            # Metric calculation
            # Get frame-wise predictions
            preds_frame = torch.argmax(logits, dim=2).cpu().numpy()  # (B, T)
            targets_cpu = targets.cpu().numpy()  # (B, T)

            for i in range(len(targets)):
                # Decode Prediction
                # Slice by length to avoid processing padding
                length = lengths[i].item()
                p_seq = preds_frame[i, :length]
                decoded_pred = rle_decode(p_seq)
                all_preds.append(decoded_pred)

                # Decode Target
                t_seq = targets_cpu[i, :length]
                decoded_target = decode_target_seq(t_seq)
                all_targets.append(decoded_target)

    avg_loss = total_loss / count if count > 0 else 0.0
    error_rate = compute_error_rate(all_preds, all_targets)

    return avg_loss, error_rate


def train_model(
    num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE, debug=False
):
    """
    Main training loop.
    Args:
        num_epochs (int): Number of epochs to train.
        batch_size (int): Batch size.
        debug (bool): If True, runs for a few steps/epochs for debugging.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # Initialize Model
    model = GCA_IIN().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Loss Function
    # We do not ignore index 0; we want the model to learn background.
    # We apply class weights if necessary, but prompt suggests simple CE with smoothing.
    # Background weight is handled via weight tensor if we strictly follow config,
    # but Config.BACKGROUND_WEIGHT is defined. Let's use it.
    class_weights = torch.ones(Config.NUM_CLASSES).to(device)
    class_weights[Config.BACKGROUND_CLASS_ID] = Config.BACKGROUND_WEIGHT

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING, reduction="mean"
    )

    best_error_rate = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_error = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Error Rate: {val_error:.10f}"
        )

        # Checkpoint & Early Stopping
        if val_error < best_error_rate:
            best_error_rate = val_error
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"New best model saved with Error Rate: {best_error_rate:.10f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        if debug and epoch >= 2:
            break

    print(f"Training complete. Best Validation Error Rate: {best_error_rate:.10f}")
    return best_model_path


def predict_test(model_path=None, batch_size=Config.BATCH_SIZE):
    """
    Runs inference on the test set and generates the submission file.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    if model_path is None:
        model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Load Data
    _, _, test_loader = get_dataloaders(batch_size=batch_size)

    # Load Model
    model = GCA_IIN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []

    print("Starting inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            lengths = batch["lengths"].to(device)
            sample_ids = batch["sample_ids"]

            logits = model(skeleton, audio, lengths)
            preds_frame = torch.argmax(logits, dim=2).cpu().numpy()

            for i, sample_id in enumerate(sample_ids):
                length = lengths[i].item()
                p_seq = preds_frame[i, :length]

                # Decode
                decoded_seq = rle_decode(p_seq)

                # Format: SessionID,label1,label2,...
                label_str = ",".join(map(str, decoded_seq))
                results.append(f"{sample_id},{label_str}")

    # Save Submission
    submission_path = Config.SUBMISSION_PATH
    with open(submission_path, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Submission saved to {submission_path}")
