import os
import time
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import LidarDataset
from library.detector import TwoStagePointPillars


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, num_epochs):
    """
    Executes one epoch of training.
    """
    model.train()

    running_loss_total = 0.0
    running_loss_hm = 0.0
    running_loss_box = 0.0
    running_loss_refine = 0.0

    num_batches = len(dataloader)
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, list):
                # gt_boxes is a list of tensors
                batch[key] = [
                    v.to(device) if isinstance(v, torch.Tensor) else v for v in value
                ]

        optimizer.zero_grad()

        # Forward pass (mode='train' returns loss dict)
        loss_dict = model(batch, mode="train")

        loss_total = loss_dict["total_loss"]
        loss_hm = loss_dict["loss_hm"]
        loss_box = loss_dict["loss_box"]
        loss_refine = loss_dict["loss_refine"]

        # Backward pass
        loss_total.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_NORM_CLIP)

        optimizer.step()
        scheduler.step()

        # Accumulate stats
        running_loss_total += loss_total.item()
        running_loss_hm += loss_hm.item()
        running_loss_box += loss_box.item()
        running_loss_refine += loss_refine.item()

    avg_total = running_loss_total / num_batches
    avg_hm = running_loss_hm / num_batches
    avg_box = running_loss_box / num_batches
    avg_refine = running_loss_refine / num_batches

    elapsed = time.time() - start_time
    print(
        f"Epoch [{epoch+1}/{num_epochs}] Train Time: {elapsed:.2f}s | "
        f"Total: {avg_total:.6f} | HM: {avg_hm:.6f} | Box: {avg_box:.6f} | Refine: {avg_refine:.6f}"
    )

    return avg_total


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set using loss metrics.
    """
    model.eval()

    val_loss_total = 0.0
    val_loss_hm = 0.0
    val_loss_box = 0.0
    val_loss_refine = 0.0

    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
                elif isinstance(value, list):
                    batch[key] = [
                        v.to(device) if isinstance(v, torch.Tensor) else v
                        for v in value
                    ]

            # Forward pass in train mode to get validation losses
            loss_dict = model(batch, mode="train")

            val_loss_total += loss_dict["total_loss"].item()
            val_loss_hm += loss_dict["loss_hm"].item()
            val_loss_box += loss_dict["loss_box"].item()
            val_loss_refine += loss_dict["loss_refine"].item()

    avg_total = val_loss_total / num_batches
    avg_hm = val_loss_hm / num_batches
    avg_box = val_loss_box / num_batches
    avg_refine = val_loss_refine / num_batches

    print(
        f"Validation Metrics       | "
        f"Total: {avg_total} | HM: {avg_hm} | Box: {avg_box} | Refine: {avg_refine}"
    )

    return avg_total


def run_training(max_epochs=None, patience=5):
    """
    Main driver function for training.
    """
    # 1. Reproducibility
    Config.set_seed(Config.SEED)

    # 2. Configuration
    device = Config.DEVICE
    epochs = max_epochs if max_epochs is not None else Config.NUM_EPOCHS
    batch_size = Config.BATCH_SIZE
    num_workers = Config.NUM_WORKERS

    print(f"Starting training on {device} for {epochs} epochs...")
    print(f"Batch Size: {batch_size}, Workers: {num_workers}")

    # 3. Data Loading
    # Note: LidarDataset handles caching internally via _load_transforms
    train_dataset = LidarDataset(split="train", load_cached_data=True)
    val_dataset = LidarDataset(split="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=LidarDataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=LidarDataset.collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 4. Model Initialization
    model = TwoStagePointPillars()
    model.to(device)

    # 5. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 6. Training Loop with Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, epochs
        )

        # Validate
        val_loss = validate(model, val_loader, device)

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss} to {val_loss}. Saving checkpoint..."
            )
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")
    print(f"Best model saved to: {Config.CHECKPOINT_PATH}")
