import os
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config, seed_everything
from library.dataset import SIIMDataset, get_transforms
from library.model import MultiTaskUNet
from library.loss import MultiTaskLoss


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss_total = 0.0
    running_loss_cls = 0.0
    running_loss_seg = 0.0

    for batch_idx, batch in enumerate(loader):
        # Move data to device
        images = batch["image"].to(device)
        cls_targets = batch["label"].to(device)
        mask_targets = batch["mask"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        cls_logits, mask_logits = model(images)

        # Calculate loss
        loss, metrics = criterion(cls_logits, mask_logits, cls_targets, mask_targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Accumulate metrics
        running_loss_total += metrics["loss_total"]
        running_loss_cls += metrics["loss_cls"]
        running_loss_seg += metrics["loss_seg"]

    # Calculate averages
    dataset_size = len(loader)
    avg_loss_total = running_loss_total / dataset_size
    avg_loss_cls = running_loss_cls / dataset_size
    avg_loss_seg = running_loss_seg / dataset_size

    return {
        "loss_total": avg_loss_total,
        "loss_cls": avg_loss_cls,
        "loss_seg": avg_loss_seg,
    }


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss_total = 0.0
    running_loss_cls = 0.0
    running_loss_seg = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Move data to device
            images = batch["image"].to(device)
            cls_targets = batch["label"].to(device)
            mask_targets = batch["mask"].to(device)

            # Forward pass
            cls_logits, mask_logits = model(images)

            # Calculate loss
            loss, metrics = criterion(
                cls_logits, mask_logits, cls_targets, mask_targets
            )

            # Accumulate metrics
            running_loss_total += metrics["loss_total"]
            running_loss_cls += metrics["loss_cls"]
            running_loss_seg += metrics["loss_seg"]

    # Calculate averages
    dataset_size = len(loader)
    avg_loss_total = running_loss_total / dataset_size
    avg_loss_cls = running_loss_cls / dataset_size
    avg_loss_seg = running_loss_seg / dataset_size

    return {
        "val_loss_total": avg_loss_total,
        "val_loss_cls": avg_loss_cls,
        "val_loss_seg": avg_loss_seg,
    }


def run_training(
    debug=Config.DEBUG,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Starting training on device: {device}")

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Debug mode: subset data
    if debug:
        print(
            f"Debug mode enabled. Subsetting data to {Config.MAX_TRAIN_SAMPLES} samples."
        )
        train_df = train_df.head(Config.MAX_TRAIN_SAMPLES)
        val_df = val_df.head(Config.MAX_VAL_SAMPLES)

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    # 3. Datasets and Loaders
    train_dataset = SIIMDataset(
        df=train_df,
        split="train",
        transform=get_transforms("train"),
        load_cached_data=load_cached_data,
    )

    val_dataset = SIIMDataset(
        df=val_df,
        split="val",
        transform=get_transforms("val"),
        load_cached_data=load_cached_data,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
    )

    # 4. Model, Optimizer, Loss
    model = MultiTaskUNet(pretrained=True).to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion = MultiTaskLoss()

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Beginning training loop...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch}/{epochs} - Time: {elapsed}s")
        print(f"Train Loss: {train_metrics['loss_total']}")
        print(f"Train Cls Loss: {train_metrics['loss_cls']}")
        print(f"Train Seg Loss: {train_metrics['loss_seg']}")
        print(f"Val Loss: {val_metrics['val_loss_total']}")
        print(f"Val Cls Loss: {val_metrics['val_loss_cls']}")
        print(f"Val Seg Loss: {val_metrics['val_loss_seg']}")

        # Early Stopping & Checkpointing
        current_val_loss = val_metrics["val_loss_total"]

        if current_val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss} to {current_val_loss}. Saving model..."
            )
            best_val_loss = current_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered. Training finished.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")
