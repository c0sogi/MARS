import os
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import AverageMeter, get_device, print_metrics
from library.data_loader import get_fold_loaders, get_test_loader
from library.model import IcebergResNet34


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    # Label smoothing factor
    epsilon = Config.LABEL_SMOOTHING

    for batch_idx, (images, angles, labels) in enumerate(loader):
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        # Apply Label Smoothing: y_new = y * (1 - eps) + 0.5 * eps
        # This assumes binary classification with targets 0 and 1
        with torch.no_grad():
            smooth_labels = labels * (1.0 - epsilon) + 0.5 * epsilon

        optimizer.zero_grad()

        logits = model(images, angles)
        loss = criterion(logits, smooth_labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate_one_epoch(model, loader, criterion, device):
    """
    Validates the model and calculates Log Loss.
    """
    model.eval()
    losses = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            logits = model(images, angles)

            # Calculate validation loss against hard labels (no smoothing)
            loss = criterion(logits, labels)
            losses.update(loss.item(), images.size(0))

            # Collect probabilities and targets for metric calculation
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Concatenate results
    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_targets).flatten()

    # Clip predictions to avoid log(0) - standard practice for Log Loss
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # Calculate Log Loss using sklearn
    metric_log_loss = log_loss(y_true, y_pred)

    return metric_log_loss


def run_fold(fold_idx, load_cached_data=True):
    """
    Runs the training pipeline for a single fold.
    """
    device = get_device()
    print(f"\n[Fold {fold_idx}] Starting training on {device}...")

    # Get DataLoaders
    train_loader, val_loader = get_fold_loaders(
        fold_idx, load_cached_data=load_cached_data
    )

    # Initialize Model
    model = IcebergResNet34()
    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # Loss Function (BCEWithLogitsLoss)
    criterion = nn.BCEWithLogitsLoss()

    # Training State
    best_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    save_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold_idx}_best.pth")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Logging
        metrics = {
            "Epoch": f"{epoch + 1}/{Config.NUM_EPOCHS}",
            "Train Loss": f"{train_loss:.6f}",
            "Val LogLoss": f"{val_loss:.6f}",
            "LR": f"{current_lr:.2e}",
            "Time": f"{elapsed:.1f}s",
        }
        print_metrics(metrics, prefix=f"Fold {fold_idx}")

        # Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"[Fold {fold_idx}] Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"[Fold {fold_idx}] Best Log Loss: {best_loss:.6f}")
    return save_path


def train_all_folds():
    """
    Sequentially trains all folds defined in Config.
    """
    model_paths = []
    for fold in range(Config.NUM_FOLDS):
        path = run_fold(fold)
        model_paths.append(path)
    return model_paths


def inference_tta(model, loader, device):
    """
    Performs inference with Test Time Augmentation (Original + HFlip + VFlip).
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, angles, img_ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            # 1. Original
            logits_orig = model(images, angles)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip
            images_hflip = torch.flip(images, dims=[3])  # (B, C, H, W) -> flip W
            logits_hflip = model(images_hflip, angles)
            probs_hflip = torch.sigmoid(logits_hflip)

            # 3. Vertical Flip
            images_vflip = torch.flip(images, dims=[2])  # (B, C, H, W) -> flip H
            logits_vflip = model(images_vflip, angles)
            probs_vflip = torch.sigmoid(logits_vflip)

            # Average probabilities
            avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0

            all_preds.append(avg_probs.cpu().numpy())
            all_ids.extend(img_ids)

    return np.concatenate(all_preds).flatten(), all_ids


def generate_submission(model_paths):
    """
    Generates submission file by averaging predictions from all fold models.
    """
    print("\nStarting Submission Generation...")
    device = get_device()
    test_loader = get_test_loader(load_cached_data=True)

    ensemble_preds = None
    test_ids = None

    for i, path in enumerate(model_paths):
        print(f"Loading model from {path}...")
        model = IcebergResNet34()
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)

        preds, ids = inference_tta(model, test_loader, device)

        if ensemble_preds is None:
            ensemble_preds = preds
            test_ids = ids
        else:
            ensemble_preds += preds

    # Average over folds
    ensemble_preds /= len(model_paths)

    # Create DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": ensemble_preds})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Done.")
