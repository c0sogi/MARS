import os
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from library.config import CFG
from library.dataset import UWMGIDataset, get_transforms, process_25d_dataframe
from library.model import build_model
from library.losses import CompositeLoss
from library.utils import rle_encode, compute_metrics


def train_one_epoch(
    model, optimizer, scheduler, dataloader, device, epoch, scaler, criterion
):
    """
    Trains the model for one epoch.
    """
    model.train()

    dataset_size = 0
    running_loss = 0.0

    # Iterate over batches
    # Using tqdm for progress tracking is allowed but we keep it minimal/silent if needed
    # Here we iterate directly
    for step, (images, masks, ids) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        with autocast(enabled=CFG.mixed_precision):
            y_pred = model(images)
            if CFG.deep_supervision and isinstance(y_pred, (list, tuple)):
                loss = 0
                for logits in y_pred:
                    loss += criterion(logits, masks)
                loss /= len(y_pred)
            else:
                loss = criterion(y_pred, masks)

        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Scheduler step (if per batch, but usually CosineAnnealing is per epoch)
        # CFG.T_max = epochs implies per-epoch stepping, but let's check standard usage.
        # Standard PyTorch CosineAnnealingLR is stepped per epoch.
        # If using OneCycleLR, it's per batch. We stick to per epoch for CosineAnnealing.

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size

    # Step scheduler at the end of epoch
    if scheduler is not None:
        scheduler.step()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, df_valid):
    """
    Validates the model by reconstructing 3D volumes and computing competition metrics.
    """
    model.eval()

    pred_rows = []
    classes = ["large_bowel", "small_bowel", "stomach"]

    # Inference loop
    for step, (images, ids) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float)
        batch_size = images.size(0)

        with autocast(enabled=CFG.mixed_precision):
            y_pred = model(images)

        if CFG.deep_supervision and isinstance(y_pred, (list, tuple)):
            y_pred = y_pred[0]

        # Apply sigmoid and threshold
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred > CFG.mask_threshold).float()

        # Move to CPU for RLE encoding
        y_pred = y_pred.cpu().numpy().astype(np.uint8)

        # Process batch
        for i in range(batch_size):
            sample_id = ids[i]
            sample_pred = y_pred[i]  # (C, H, W)

            # For each class, encode RLE
            for class_idx, class_name in enumerate(classes):
                mask = sample_pred[class_idx]
                rle = rle_encode(mask)
                pred_rows.append(
                    {"id": sample_id, "class": class_name, "predicted": rle}
                )

    # Create prediction DataFrame
    df_pred = pd.DataFrame(pred_rows)

    # Compute Metrics
    # df_valid contains the ground truth metadata
    metrics = compute_metrics(df_pred, df_valid)

    return metrics


def train():
    """
    Main training pipeline.
    """
    # 1. Setup
    CFG.setup()
    device = CFG.device

    # 2. Data Preparation
    print(f"Loading metadata from {CFG.meta_dir}...")
    df_train = pd.read_csv(CFG.train_csv)
    df_val = pd.read_csv(CFG.val_csv)

    # Debug mode
    if CFG.debug:
        print(f"Debug mode: using {CFG.debug_size} samples.")
        df_train = df_train.iloc[: CFG.debug_size]
        df_val = df_val.iloc[: CFG.debug_size]

    # Process 2.5D Data
    print("Processing 2.5D context...")
    df_train = process_25d_dataframe(
        df_train, split_name="train", load_cached_data=True
    )
    df_val = process_25d_dataframe(df_val, split_name="val", load_cached_data=True)

    # Create Datasets
    train_dataset = UWMGIDataset(
        df_train, label=True, transforms=get_transforms(data="train")
    )
    valid_dataset = UWMGIDataset(
        df_val,
        label=False,  # We don't need masks in __getitem__ for valid loop inference, we use df_val for GT
        transforms=get_transforms(data="valid"),
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Model Initialization
    print(f"Building model: {CFG.model_arch} with {CFG.backbone}...")
    model = build_model()
    model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr
    )
    criterion = CompositeLoss().to(device)
    scaler = GradScaler(enabled=CFG.mixed_precision)

    # 5. Training Loop
    best_score = -np.inf

    print(f"Starting training for {CFG.epochs} epochs...")

    for epoch in range(CFG.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch, scaler, criterion
        )

        # Validate
        val_metrics = valid_one_epoch(model, valid_loader, device, df_val)
        val_score = val_metrics["score"]
        val_dice = val_metrics["dice"]
        val_hd = val_metrics["hausdorff"]

        elapsed = time.time() - start_time

        # Logging
        print(
            f"Epoch {epoch+1}/{CFG.epochs} [{elapsed:.0f}s]: "
            f"Loss: {train_loss:.4f} | "
            f"Valid Score: {val_score} | "
            f"Dice: {val_dice} | "
            f"HD: {val_hd}"
        )

        # Checkpointing
        # Save Last
        torch.save(
            model.state_dict(), os.path.join(CFG.checkpoint_dir, "last_model.pth")
        )

        # Save Best
        if val_score > best_score:
            print(f"Score Improved ({best_score} -> {val_score}). Saving best model...")
            best_score = val_score
            torch.save(
                model.state_dict(), os.path.join(CFG.checkpoint_dir, "best_model.pth")
            )

        # Memory Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Training complete. Best Score: {best_score}")


if __name__ == "__main__":
    train()
