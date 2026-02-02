import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import set_seed, AverageMeter, kl_divergence_score
from library.data import get_loaders
from library.models import TriViewNet


def train_one_epoch(loader, model, optimizer, scheduler, scaler, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()
    # KLDivLoss expects log-probabilities as input
    criterion = nn.KLDivLoss(reduction="batchmean")

    for batch_idx, batch in enumerate(loader):
        # Move data to device
        micro = batch["micro"].to(device, non_blocking=True)
        meso = batch["meso"].to(device, non_blocking=True)
        macro = batch["macro"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            # Forward pass
            logits = model(micro, meso, macro)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, targets)

        # Backward pass with scaler
        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer and Scheduler steps
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.update(loss.item(), micro.size(0))

    return losses.avg


def validate(loader, model, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    kl_scores = AverageMeter()
    criterion = nn.KLDivLoss(reduction="batchmean")

    with torch.no_grad():
        for batch in loader:
            micro = batch["micro"].to(device, non_blocking=True)
            meso = batch["meso"].to(device, non_blocking=True)
            macro = batch["macro"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            # Forward pass
            logits = model(micro, meso, macro)
            log_probs = F.log_softmax(logits, dim=1)
            probs = F.softmax(logits, dim=1)

            # Compute metrics
            loss = criterion(log_probs, targets)
            kl = kl_divergence_score(targets, probs)

            losses.update(loss.item(), micro.size(0))
            kl_scores.update(kl, micro.size(0))

    return losses.avg, kl_scores.avg


def train(debug=False, load_cached_data=False, epochs=Config.EPOCHS):
    """
    Main training loop with Early Stopping and Checkpointing.

    Args:
        debug (bool): If True, runs on a small subset of data.
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
        epochs (int): Number of training epochs.
    """
    set_seed(Config.SEED)

    # Ensure checkpoint directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Initialize DataLoaders
    train_loader, val_loader, _ = get_loaders(
        debug=debug, load_cached_data=load_cached_data
    )

    # Initialize Model
    print(f"Initializing TriViewNet on {Config.DEVICE}...")
    model = TriViewNet(num_classes=Config.NUM_CLASSES, pretrained=True)
    model.to(Config.DEVICE)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (OneCycleLR)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # Mixed Precision Scaler
    scaler = GradScaler()

    # Training State
    best_score = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")
    print(f"Batch Size: {Config.BATCH_SIZE}, Training Steps: {steps_per_epoch}")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, model, optimizer, scheduler, scaler, Config.DEVICE, epoch
        )

        # Validate
        val_loss, val_kl = validate(val_loader, model, Config.DEVICE)

        elapsed = time.time() - start_time

        # Log Metrics (Full Precision)
        print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val KL Score: {val_kl}")

        # Checkpointing & Early Stopping
        if val_kl < best_score:
            best_score = val_kl
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation KL Score: {best_score}")
