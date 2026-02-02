import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import (
    set_seed,
    AverageMeter,
    KLDivLossWithLogits,
    kl_divergence_score,
)
from library.dataset import get_dataloaders
from library.model import HybridModel


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler):
    """
    Performs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for eeg, spec, targets in loader:
        eeg = eeg.to(device, non_blocking=True)
        spec = spec.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=True):
            logits = model(eeg, spec)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.update(loss.item(), eeg.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and KL divergence score.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    gts = []

    with torch.no_grad():
        for eeg, spec, targets in loader:
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(enabled=True):
                logits = model(eeg, spec)
                loss = criterion(logits, targets)

            losses.update(loss.item(), eeg.size(0))

            # Apply softmax to get probabilities for metric calculation
            probs = torch.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())
            gts.append(targets.cpu().numpy())

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    score = kl_divergence_score(gts, preds)

    return losses.avg, score


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for eeg, spec in loader:
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)

            with autocast(enabled=True):
                logits = model(eeg, spec)

            probs = torch.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())

    return np.concatenate(preds)


def run_training(load_cached_data=True):
    """
    Main execution function for training and inference.
    """
    # Initialize Config
    config = Config()

    # Set Seed
    set_seed(config.seed)

    # Create necessary directories
    os.makedirs(config.working_dir, exist_ok=True)
    os.makedirs(config.submission_dir, exist_ok=True)

    # Get DataLoaders
    loaders = get_dataloaders(config, load_cached_data=load_cached_data)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders.get("test", None)

    # Initialize Model
    device = torch.device(config.device)
    model = HybridModel(config)
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Scheduler (OneCycleLR)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.lr,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=config.pct_start,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor,
    )

    # Loss & Scaler
    criterion = KLDivLossWithLogits()
    scaler = GradScaler(enabled=config.use_amp)

    # Training Loop
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(config.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler
        )
        val_loss, val_score = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Time: {elapsed}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Score (KL): {val_score}"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), config.model_save_path)
            print(f"New best model saved to {config.model_save_path}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {config.patience}")

        if patience_counter >= config.patience:
            print("Early stopping triggered.")
            break

    # Inference on Test Set
    if test_loader is not None:
        print("Loading best model for inference...")
        model.load_state_dict(torch.load(config.model_save_path, map_location=device))

        print("Generating predictions...")
        test_preds = predict(model, test_loader, device)

        # Create Submission File
        # Load test metadata to get eeg_ids
        df_test = pd.read_csv(config.test_csv)

        submission = pd.DataFrame(test_preds, columns=config.vote_cols)
        submission.insert(0, "eeg_id", df_test["eeg_id"])

        submission.to_csv(config.submission_path, index=False)
        print(f"Submission saved to {config.submission_path}")
