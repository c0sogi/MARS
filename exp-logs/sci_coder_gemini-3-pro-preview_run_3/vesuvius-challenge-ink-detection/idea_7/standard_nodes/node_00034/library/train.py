import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.model import InkDetector
from library.utils import dice_loss, fbeta_score, predict_tiled
from library.data import get_dataloaders, load_fragment, get_global_stats


def train_one_epoch(model, loader, optimizer, device):
    """
    Trains the model for one epoch using Balanced Loss (BCE + Dice).
    """
    model.train()
    running_loss = 0.0

    # BCEWithLogitsLoss combines Sigmoid and BCE for numerical stability
    bce_criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(data)

        # Calculate losses
        # Target shape is (B, 1, H, W)
        loss_bce = bce_criterion(logits, target)
        loss_dice = dice_loss(logits, target)

        # Balanced Loss
        loss = loss_bce + loss_dice

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def optimize_threshold(preds, targets):
    """
    Finds the best threshold that maximizes F0.5 score.
    preds: np.array of probabilities (0-1)
    targets: np.array of binary labels (0 or 1)
    """
    best_threshold = 0.5
    best_score = 0.0

    # Convert to tensor for faster computation on GPU if available,
    # but usually validation set is flattened and large, CPU is safer for memory
    # unless we chunk it. Given 220GB RAM, CPU numpy is fine and simple.
    # However, fbeta_score in utils expects tensors.

    preds_tensor = torch.from_numpy(preds)
    targets_tensor = torch.from_numpy(targets)

    # We treat the probabilities as "logits" for fbeta_score if we didn't apply sigmoid.
    # But predict_tiled returns probabilities (sigmoid applied).
    # library.utils.fbeta_score applies sigmoid internally to logits.
    # To use fbeta_score with probabilities, we need to inverse sigmoid or modify logic.
    # Easier approach: Implement simple F0.5 calculation here for numpy arrays
    # to avoid inverse sigmoid instability.

    steps = Config.THRESHOLD_SEARCH_STEPS
    thresholds = np.linspace(0.1, 0.9, steps)

    beta = 0.5

    for th in thresholds:
        # Vectorized numpy calculation
        pred_bin = (preds > th).astype(np.float32)

        tp = (pred_bin * targets).sum()
        fp = (pred_bin * (1 - targets)).sum()
        fn = ((1 - pred_bin) * targets).sum()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        score = (
            (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-6)
        )

        if score > best_score:
            best_score = score
            best_threshold = th

    return best_score, best_threshold


def validate(model, val_metadata_path, mean, std, device, load_cached_data=True):
    """
    Performs deterministic validation on the full fragment using tiled inference.
    """
    df_val = pd.read_csv(val_metadata_path)

    all_preds = []
    all_targets = []

    # Ensure cache dir exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    for _, row in df_val.iterrows():
        # Load full fragment data
        volume, mask, label = load_fragment(
            row, Config.WORKING_DIR, load_cached_data=load_cached_data
        )

        if label is None:
            continue

        # Run inference
        # predict_tiled handles normalization using provided mean/std
        pred_map = predict_tiled(
            model,
            volume,
            patch_size=Config.PATCH_SIZE,
            stride=Config.STRIDE,
            device=device,
            mean=mean,
            std=std,
        )

        # Flatten and select only valid pixels defined by the mask
        mask_bool = mask > 0

        valid_preds = pred_map[mask_bool]
        valid_targets = label[mask_bool]

        all_preds.append(valid_preds)
        all_targets.append(valid_targets)

    if not all_preds:
        return 0.0, 0.5

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Optimize threshold
    best_f05, best_th = optimize_threshold(all_preds, all_targets)

    return best_f05, best_th


def train_model(load_cached_data=True, num_train_samples=4000):
    """
    Main training loop with Early Stopping.
    """
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Data Preparation
    # Get global stats for normalization
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    mean, std = get_global_stats(
        df_train, Config.WORKING_DIR, load_cached_data=load_cached_data
    )

    # Get Loaders
    train_loader, _ = get_dataloaders(
        train_metadata_path=Config.TRAIN_METADATA,
        val_metadata_path=Config.VAL_METADATA,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
        num_train_samples=num_train_samples,
    )

    # 2. Model Setup
    model = InkDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 3. Training Loop
    best_f05 = 0.0
    best_threshold = 0.5
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        # We pass the path to metadata so validate can load full fragments
        val_f05, val_th = validate(
            model,
            Config.VAL_METADATA,
            mean,
            std,
            device,
            load_cached_data=load_cached_data,
        )

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val F0.5: {val_f05:.10f} - "
            f"Best Thresh: {val_th:.4f}"
        )

        # Early Stopping & Checkpointing
        if val_f05 > best_f05:
            best_f05 = val_f05
            best_threshold = val_th
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), Config.CHECKPOINT_DIR / "best_model.pth")

            # Save threshold
            with open(Config.WORKING_DIR / "threshold.txt", "w") as f:
                f.write(str(best_threshold))

            print(f"  -> New best model saved! F0.5: {best_f05:.10f}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(
        f"Training complete. Best Val F0.5: {best_f05:.10f} at Threshold: {best_threshold:.4f}"
    )
    return best_f05
