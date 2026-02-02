import os
import gc
import time
import random
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast

from library.config import (
    PATHS,
    TRAINING_PARAMS,
    DEVICE,
    SEED,
    NUM_WORKERS,
    SPECIALIST_SETTINGS,
)
from library.dataset import InkDataset
from library.model import get_model
from library.loss import BCEDiceLoss


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_fbeta(preds, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Calculates the F-beta score for binary segmentation.

    Args:
        preds (torch.Tensor): Raw logits or probabilities.
        targets (torch.Tensor): Binary ground truth.
        beta (float): Beta value for F-score (0.5 weights precision higher).
        threshold (float): Threshold for binarizing predictions.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        tuple: (fbeta_score, precision, recall)
    """
    # Apply sigmoid if logits (assuming input might be logits, but usually we pass probs or logits)
    # Here we assume logits are passed, so we apply sigmoid.
    preds = torch.sigmoid(preds)

    # Binarize
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > threshold).float()

    # Flatten
    preds_flat = preds_bin.view(-1)
    targets_flat = targets_bin.view(-1)

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    beta_sq = beta**2
    fbeta = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + epsilon)
    )

    return fbeta.item(), precision.item(), recall.item()


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with autocast(enabled=TRAINING_PARAMS["use_amp"]):
            logits = model(images)
            loss = criterion(logits, labels)

        # Backward Pass
        if scaler is not None:
            scaler.scale(loss).backward()
            if TRAINING_PARAMS["clip_grad"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), TRAINING_PARAMS["clip_grad"]
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if TRAINING_PARAMS["clip_grad"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), TRAINING_PARAMS["clip_grad"]
                )
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    """
    Executes validation and computes global F0.5 score.
    """
    model.eval()
    running_loss = 0.0

    # Accumulators for global metrics
    tp_total = 0
    fp_total = 0
    fn_total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            with autocast(enabled=TRAINING_PARAMS["use_amp"]):
                logits = model(images)
                loss = criterion(logits, labels)

            running_loss += loss.item()

            # Metric Calculation (Accumulate counts)
            probs = torch.sigmoid(logits)
            preds_bin = (probs > 0.5).float()

            # Flatten
            preds_flat = preds_bin.view(-1)
            targets_flat = labels.view(-1)

            tp = (preds_flat * targets_flat).sum().item()
            fp = (preds_flat * (1 - targets_flat)).sum().item()
            fn = ((1 - preds_flat) * targets_flat).sum().item()

            tp_total += tp
            fp_total += fp
            fn_total += fn

    avg_loss = running_loss / len(loader)

    # Compute Global F0.5
    epsilon = 1e-7
    precision = tp_total / (tp_total + fp_total + epsilon)
    recall = tp_total / (tp_total + fn_total + epsilon)
    beta = 0.5
    beta_sq = beta**2
    f05_score = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + epsilon)
    )

    return avg_loss, f05_score


def clear_cache_for_specialist(specialist_mode):
    """
    Clears cached .npy files for the specific specialist configuration to force regeneration.
    """
    if specialist_mode not in SPECIALIST_SETTINGS:
        return

    settings = SPECIALIST_SETTINGS[specialist_mode]
    z_start = settings["z_start"]
    z_end = settings["z_end"]

    # Pattern: frag_{id}_slab_{z_start}_{z_end}.npy
    pattern = f"frag_*_slab_{z_start}_{z_end}.npy"
    search_path = os.path.join(PATHS.WORKING_DIR, pattern)

    files = glob.glob(search_path)
    if files:
        print(
            f"Clearing {len(files)} cached files for specialist '{specialist_mode}' to force regeneration."
        )
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print(f"Error removing {f}: {e}")


def run_specialist_training(specialist_mode, load_cached_data=True):
    """
    Main function to train a specialist model.

    Args:
        specialist_mode (str): 'High', 'Mid', or 'Low'.
        load_cached_data (bool): If False, clears existing cache for this mode before starting.
    """
    print(f"\n{'='*40}")
    print(f"Starting Training for Specialist: {specialist_mode}")
    print(f"{'='*40}")

    set_seed(SEED)

    # 1. Cache Management
    # Since InkDataset hardcodes load_cached_data=True, we manage it by deleting files if requested.
    if not load_cached_data:
        clear_cache_for_specialist(specialist_mode)

    # 2. Prepare Data
    print("Loading Metadata...")
    df_train = pd.read_csv(PATHS.TRAIN_METADATA)
    df_val = pd.read_csv(PATHS.VAL_METADATA)

    # Debug mode: subset data
    if TRAINING_PARAMS.get("debug", False):
        print("DEBUG MODE: Using subset of data.")
        df_train = df_train.head(16)
        df_val = df_val.head(8)

    # Initialize Datasets
    # Note: Dataset initialization triggers cache generation/loading
    train_dataset = InkDataset(
        metadata=df_train, specialist_mode=specialist_mode, split="train"
    )

    val_dataset = InkDataset(
        metadata=df_val, specialist_mode=specialist_mode, split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_PARAMS["batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_PARAMS["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # 3. Model Setup
    model = get_model(TRAINING_PARAMS.get("model_config", None))
    model.to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAINING_PARAMS["learning_rate"],
        weight_decay=TRAINING_PARAMS["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAINING_PARAMS["epochs"], eta_min=TRAINING_PARAMS["min_lr"]
    )

    criterion = BCEDiceLoss()

    scaler = GradScaler() if TRAINING_PARAMS["use_amp"] else None

    # 4. Training Loop
    best_f05 = 0.0
    save_path = os.path.join(PATHS.WORKING_DIR, f"model_{specialist_mode}.pth")

    for epoch in range(1, TRAINING_PARAMS["epochs"] + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, DEVICE
        )

        # Validate
        val_loss, val_f05 = validate_one_epoch(model, val_loader, criterion, DEVICE)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{TRAINING_PARAMS['epochs']} | "
            f"Time: {elapsed:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val F0.5: {val_f05:.10f}"
        )

        # Checkpoint Saving
        if val_f05 > best_f05:
            best_f05 = val_f05
            if best_f05 >= TRAINING_PARAMS["valid_threshold"]:
                print(f"New Best F0.5: {best_f05:.10f}. Saving model to {save_path}")
                torch.save(model.state_dict(), save_path)
            else:
                print(
                    f"New Best F0.5: {best_f05:.10f}, but below threshold {TRAINING_PARAMS['valid_threshold']}. Not saving."
                )

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nTraining Finished for Specialist {specialist_mode}.")
    print(f"Best Validation F0.5: {best_f05:.10f}")
    if os.path.exists(save_path):
        print(f"Best model saved at: {save_path}")
    else:
        print("No model saved (did not exceed threshold).")
