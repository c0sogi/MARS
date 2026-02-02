import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, AverageMeter, kl_divergence_score
from library.data import load_data, EEGDataset, mixup_data
from library.model import BandAdaptiveNet


def train_one_epoch(epoch, model, train_loader, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    # KLDivLoss expects input as log-probabilities
    criterion = nn.KLDivLoss(reduction="batchmean")

    for step, ((x_eeg, x_spec), y) in enumerate(train_loader):
        x_eeg = x_eeg.to(device)
        x_spec = x_spec.to(device)
        y = y.to(device)

        # Apply MixUp
        x_eeg, x_spec, y_a, y_b, lam = mixup_data(
            x_eeg, x_spec, y, alpha=Config.MIXUP_ALPHA, device=device
        )

        optimizer.zero_grad()

        # Automatic Mixed Precision
        with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
            logits = model(x_eeg, x_spec)
            log_probs = F.log_softmax(logits, dim=1)

            # MixUp Loss
            loss = lam * criterion(log_probs, y_a) + (1 - lam) * criterion(
                log_probs, y_b
            )

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        loss_meter.update(loss.item(), x_eeg.size(0))

    return loss_meter.avg


def validate_one_epoch(epoch, model, val_loader, device):
    """
    Runs validation on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()
    criterion = nn.KLDivLoss(reduction="batchmean")

    preds = []
    targets = []

    with torch.no_grad():
        for step, ((x_eeg, x_spec), y) in enumerate(val_loader):
            x_eeg = x_eeg.to(device)
            x_spec = x_spec.to(device)
            y = y.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
                logits = model(x_eeg, x_spec)
                log_probs = F.log_softmax(logits, dim=1)
                loss = criterion(log_probs, y)

            loss_meter.update(loss.item(), x_eeg.size(0))

            # Store predictions (convert to probabilities) and targets for metric calculation
            probs = F.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())
            targets.append(y.cpu().numpy())

    # Concatenate all batches
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Calculate metric using the provided utility
    val_score = kl_divergence_score(targets, preds)

    return loss_meter.avg, val_score


def save_checkpoint(model, path):
    """
    Saves the model state dict.
    """
    torch.save(model.state_dict(), path)


def run_training(debug=Config.DEBUG):
    """
    Main driver function for training the model.
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Starting training with device: {Config.DEVICE}")

    # 1. Load Data
    # load_data handles caching internally
    train_eeg, train_spec, train_targets = load_data(
        mode="train", load_cached_data=True
    )
    val_eeg, val_spec, val_targets = load_data(mode="val", load_cached_data=True)

    if debug:
        print("DEBUG Mode: Truncating datasets.")
        train_eeg = train_eeg[:100]
        train_spec = train_spec[:100]
        train_targets = train_targets[:100]
        val_eeg = val_eeg[:100]
        val_spec = val_spec[:100]
        val_targets = val_targets[:100]

    # 2. Create Datasets and Loaders
    train_dataset = EEGDataset(train_eeg, train_spec, train_targets, mode="train")
    val_dataset = EEGDataset(val_eeg, val_spec, val_targets, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Initialize Model
    model = BandAdaptiveNet()
    model.to(Config.DEVICE)

    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    # 5. Training Loop
    best_val_loss = float("inf")
    best_val_score = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, Config.DEVICE
        )

        # Validate
        val_loss, val_score = validate_one_epoch(
            epoch, model, val_loader, Config.DEVICE
        )

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val KL Score: {val_score}"
        )

        # Early Stopping Logic (Monitoring Validation Loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_score = val_score
            patience_counter = 0
            save_checkpoint(model, best_model_path)
            print(f"  [Improved] Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"  [No Improvement] Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(
        f"Training Complete. Best Val Loss: {best_val_loss}, Best Val Score: {best_val_score}"
    )
