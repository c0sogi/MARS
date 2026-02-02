import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from library.config import CFG
from library.dataset import ContrailDataset, get_transforms
from library.model import ConvNeXtUNet
from library.losses import HybridLoss
from library.utils import set_seed


def train_model(debug=False):
    """
    Main training function for the Lightweight Pyramid-Fusion ConvNeXt U-Net.

    Args:
        debug (bool): If True, runs on a small subset of data for debugging.
    """
    # Set reproducibility
    set_seed(CFG.seed)

    # Setup directories
    os.makedirs(CFG.output_dir, exist_ok=True)

    print(f"Initializing training (Debug={debug})...")
    print(f"Device: {CFG.device}")

    # ====================================================
    # Data Loading
    # ====================================================
    train_dataset = ContrailDataset(
        metadata_path=CFG.train_metadata_path,
        split="train",
        transform=get_transforms("train"),
        debug=debug,
        load_cached_data=True,
    )

    valid_dataset = ContrailDataset(
        metadata_path=CFG.valid_metadata_path,
        split="validation",
        transform=get_transforms("validation"),
        debug=debug,
        load_cached_data=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train Size: {len(train_dataset)} | Validation Size: {len(valid_dataset)}")

    # ====================================================
    # Model Initialization
    # ====================================================
    model = ConvNeXtUNet(
        backbone_name=CFG.backbone,
        in_channels=CFG.in_channels,
        num_classes=CFG.out_channels,
        pretrained=CFG.pretrained,
    )
    model.to(CFG.device)

    # ====================================================
    # Optimization & Loss
    # ====================================================
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr
    )

    criterion = HybridLoss(bce_weight=CFG.bce_weight, dice_weight=CFG.dice_weight)

    scaler = GradScaler()

    # ====================================================
    # Training Loop
    # ====================================================
    best_score = -1.0
    patience = 5
    patience_counter = 0

    for epoch in range(CFG.epochs):
        start_time = time.time()

        # --- Train Phase ---
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(CFG.device, dtype=torch.float32)
            masks = batch["mask"].to(CFG.device, dtype=torch.float32)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_intersection = 0.0
        val_union = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                images = batch["image"].to(CFG.device, dtype=torch.float32)
                masks = batch["mask"].to(CFG.device, dtype=torch.float32)

                with autocast():
                    outputs = model(images)

                # Apply sigmoid and threshold
                preds = torch.sigmoid(outputs)
                preds = (preds > CFG.threshold).float()

                # Flatten for global dice calculation
                p_flat = preds.view(-1)
                t_flat = masks.view(-1)

                intersection = (p_flat * t_flat).sum().item()
                union = p_flat.sum().item() + t_flat.sum().item()

                val_intersection += intersection
                val_union += union

        # Compute Global Dice
        # Avoid division by zero
        if val_union == 0:
            val_score = (
                0.0 if val_intersection == 0 else 0.0
            )  # Should ideally handle empty set cases
            # If both empty, Dice is 1. But usually we have some foreground in the whole set.
            # If union is 0, it means no ground truth and no predictions.
            if val_intersection == 0 and val_union == 0:
                val_score = 1.0
        else:
            val_score = (2.0 * val_intersection) / val_union

        # Update Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{CFG.epochs} | "
            f"Time: {elapsed:.0f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Global Dice: {val_score}"
        )

        # Checkpointing
        if val_score > best_score:
            print(f"Score Improved ({best_score} -> {val_score}). Saving model...")
            best_score = val_score
            torch.save(model.state_dict(), CFG.best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs with no improvement."
            )
            break

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Training completed. Best Validation Global Dice: {best_score}")
