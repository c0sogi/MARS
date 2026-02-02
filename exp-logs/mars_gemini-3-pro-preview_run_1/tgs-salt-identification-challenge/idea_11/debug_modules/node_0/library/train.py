import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from library.config import Config
from library.dataset import SaltDataset, set_seed
from library.model import HighCapacityUNet
from library.losses import CompoundLoss
from library.utils import calc_map


def center_crop(tensor, target_h, target_w):
    """
    Center crops a tensor to the target spatial dimensions.
    Assumes tensor shape is (..., H, W).
    """
    # Get current height and width
    h, w = tensor.shape[-2:]

    # Calculate starting indices
    diff_h = (h - target_h) // 2
    diff_w = (w - target_w) // 2

    # Slice
    return tensor[..., diff_h : diff_h + target_h, diff_w : diff_w + target_w]


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for i, (images, masks, depths, _) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        # Forward pass
        logits = model(images, depths)

        # Compute loss
        loss, _ = criterion(logits, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Calculates average loss and Mean Average Precision (mAP).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks, depths, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            # Forward pass
            logits = model(images, depths)

            # Compute loss
            loss, _ = criterion(logits, masks)
            running_loss += loss.item()

            # Post-processing for mAP calculation
            # 1. Apply Sigmoid
            probs = torch.sigmoid(logits)
            # 2. Threshold at 0.5 to get binary mask
            preds = (probs > 0.5).float()

            # 3. Crop back to original 101x101 size to match competition metric requirements
            #    (Dataset pads to 128x128, so we must evaluate on the valid center region)
            preds_cropped = center_crop(preds, Config.ORIG_SIZE, Config.ORIG_SIZE)
            masks_cropped = center_crop(masks, Config.ORIG_SIZE, Config.ORIG_SIZE)

            # 4. Store for metric calculation
            #    Convert to numpy and store in list
            all_preds.extend([p.cpu().numpy() for p in preds_cropped])
            all_targets.extend([t.cpu().numpy() for t in masks_cropped])

    avg_loss = running_loss / len(loader)

    # Calculate mAP over IoU thresholds 0.5:0.95:0.05
    map_score = calc_map(all_preds, all_targets)

    return avg_loss, map_score


def train_model():
    """
    Main training routine.
    - Sets up directories and seeds.
    - Loads data.
    - Initializes Model, Loss, Optimizer, Scheduler.
    - Runs the training loop with Cyclic Cosine Annealing.
    - Saves checkpoints for each cycle and the global best model.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = SaltDataset(
        mode="train", load_cached_data=True, limit=Config.DEBUG_DATA_LIMIT
    )
    val_dataset = SaltDataset(
        mode="val", load_cached_data=True, limit=Config.DEBUG_DATA_LIMIT
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model & Training Components
    print("Initializing Model...")
    model = HighCapacityUNet().to(device)

    criterion = CompoundLoss().to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cyclic Scheduler: Restarts every EPOCHS_PER_CYCLE
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.EPOCHS_PER_CYCLE, T_mult=1, eta_min=1e-6
    )

    # 4. Training Loop
    print(
        f"Starting training for {Config.TOTAL_EPOCHS} epochs ({Config.NUM_CYCLES} cycles of {Config.EPOCHS_PER_CYCLE})."
    )

    global_best_map = 0.0
    # Track best mAP for the current cycle
    cycle_best_map = 0.0

    start_time = time.time()

    for epoch in range(Config.TOTAL_EPOCHS):
        epoch_start = time.time()

        # Determine current cycle (0-indexed)
        current_cycle = epoch // Config.EPOCHS_PER_CYCLE

        # Reset cycle best tracker at the start of a new cycle
        if epoch % Config.EPOCHS_PER_CYCLE == 0:
            cycle_best_map = 0.0
            print(f"\n--- Starting Cycle {current_cycle + 1} ---")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_map = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Checkpointing Logic
        # 1. Global Best
        if val_map > global_best_map:
            global_best_map = val_map
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )

        # 2. Cycle Best
        if val_map > cycle_best_map:
            cycle_best_map = val_map
            save_name = f"best_cycle_{current_cycle + 1}.pth"
            torch.save(
                model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, save_name)
            )

        # Logging
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{Config.TOTAL_EPOCHS} [Cycle {current_cycle+1}] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val mAP: {val_map:.10f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best Global Validation mAP: {global_best_map:.10f}")
