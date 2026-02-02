import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time

from library.config import Config
from library.dataset import VinBigDataset
from library.model import SpatiallyAwareCenterNet
from library.loss import CenterNetLoss
from library.utils import seed_everything, get_logger

# Initialize logger
logger = get_logger("Training")


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()

    running_loss = 0.0
    running_hm_loss = 0.0
    running_wh_loss = 0.0
    running_reg_loss = 0.0
    running_global_loss = 0.0

    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Input image is at batch['image']
        outputs = model(batch["image"])

        # Calculate loss
        loss, stats = criterion(outputs, batch)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        # Accumulate metrics
        running_loss += stats["loss"]
        running_hm_loss += stats["hm_loss"]
        running_wh_loss += stats["wh_loss"]
        running_reg_loss += stats["reg_loss"]
        running_global_loss += stats["global_loss"]

    # Calculate averages
    avg_loss = running_loss / num_batches
    avg_hm_loss = running_hm_loss / num_batches
    avg_wh_loss = running_wh_loss / num_batches
    avg_reg_loss = running_reg_loss / num_batches
    avg_global_loss = running_global_loss / num_batches

    return {
        "loss": avg_loss,
        "hm_loss": avg_hm_loss,
        "wh_loss": avg_wh_loss,
        "reg_loss": avg_reg_loss,
        "global_loss": avg_global_loss,
    }


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    running_loss = 0.0
    running_hm_loss = 0.0
    running_wh_loss = 0.0
    running_reg_loss = 0.0
    running_global_loss = 0.0

    num_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Forward pass
            outputs = model(batch["image"])

            # Calculate loss
            loss, stats = criterion(outputs, batch)

            # Accumulate metrics
            running_loss += stats["loss"]
            running_hm_loss += stats["hm_loss"]
            running_wh_loss += stats["wh_loss"]
            running_reg_loss += stats["reg_loss"]
            running_global_loss += stats["global_loss"]

    # Calculate averages
    avg_loss = running_loss / num_batches
    avg_hm_loss = running_hm_loss / num_batches
    avg_wh_loss = running_wh_loss / num_batches
    avg_reg_loss = running_reg_loss / num_batches
    avg_global_loss = running_global_loss / num_batches

    return {
        "loss": avg_loss,
        "hm_loss": avg_hm_loss,
        "wh_loss": avg_wh_loss,
        "reg_loss": avg_reg_loss,
        "global_loss": avg_global_loss,
    }


def run_training():
    """
    Main execution function for the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # 2. Data Preparation
    logger.info("Initializing Datasets...")
    train_dataset = VinBigDataset(
        csv_path=Config.TRAIN_META, mode="train", load_cached_data=True
    )
    val_dataset = VinBigDataset(
        csv_path=Config.VAL_META, mode="val", load_cached_data=True
    )

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

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # 3. Model Initialization
    logger.info(f"Initializing Model: {Config.BACKBONE}")
    model = SpatiallyAwareCenterNet(
        backbone_name=Config.BACKBONE, num_classes=Config.NUM_CLASSES, pretrained=True
    )
    model = model.to(device)

    # 4. Loss, Optimizer, Scheduler
    criterion = CenterNetLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    best_val_loss = float("inf")

    logger.info("Starting Training Loop...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Step Scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        elapsed = time.time() - start_time

        # Logging
        logger.info(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.2f}s | LR: {current_lr:.2e}"
        )
        logger.info(
            f"  Train Loss: {train_metrics['loss']} (HM: {train_metrics['hm_loss']}, WH: {train_metrics['wh_loss']}, Reg: {train_metrics['reg_loss']}, Global: {train_metrics['global_loss']})"
        )
        logger.info(
            f"  Val Loss:   {val_metrics['loss']} (HM: {val_metrics['hm_loss']}, WH: {val_metrics['wh_loss']}, Reg: {val_metrics['reg_loss']}, Global: {val_metrics['global_loss']})"
        )

        # Checkpointing
        # Save Last
        last_path = os.path.join(Config.CHECKPOINT_DIR, "last_model.pth")
        torch.save(model.state_dict(), last_path)

        # Save Best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            logger.info(f"  New Best Model Saved! Loss: {best_val_loss}")

    logger.info("Training Completed.")
    logger.info(f"Best Validation Loss: {best_val_loss}")
