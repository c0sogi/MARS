import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from library import config
from library import utils
from library import loss
from library import dataset
from library import model as model_lib


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Handles the training of a single epoch.
    """
    model.train()
    metric_monitor = utils.MetricMonitor()

    # Iterate over batches
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)

        # Calculate loss
        loss_val = criterion(logits, masks)

        # Backward pass
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()

        # Update metrics
        metric_monitor.update("Loss", loss_val.item())

    return metric_monitor.get_avg("Loss")


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using Global Dice coefficient.
    """
    model.eval()
    metric_monitor = utils.MetricMonitor()

    # Accumulators for Global Dice
    intersection_sum = 0.0
    union_sum = 0.0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images)

            # Calculate validation loss for monitoring
            loss_val = criterion(logits, masks)
            metric_monitor.update("Loss", loss_val.item())

            # Calculate Global Dice components
            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits)
            preds = (probs > config.THRESHOLD).float()

            # Flatten for calculation
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            intersection = (preds_flat * masks_flat).sum().item()
            pred_sum = preds_flat.sum().item()
            mask_sum = masks_flat.sum().item()

            intersection_sum += intersection
            union_sum += pred_sum + mask_sum

    # Compute Global Dice
    # Avoid division by zero
    if union_sum == 0:
        global_dice = 0.0
    else:
        global_dice = (2.0 * intersection_sum) / union_sum

    return metric_monitor.get_avg("Loss"), global_dice


def run_training(debug=False, epochs=config.EPOCHS):
    """
    Main function to run the training pipeline.

    Args:
        debug (bool): If True, runs on a small subset of data.
        epochs (int): Number of epochs to train.
    """
    # 1. Setup
    utils.seed_everything(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = config.DEVICE

    print(f"Starting training run (Debug={debug})...")
    print(f"Device: {device}")
    print(f"Output Directory: {config.OUTPUT_DIR}")

    # 2. Data Loading
    train_loader = dataset.get_dataloader(
        stage="train",
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        debug=debug,
    )

    val_loader = dataset.get_dataloader(
        stage="validation",
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        debug=debug,
    )

    # 3. Model Initialization
    model = model_lib.ConvNeXtUNet(
        backbone_name=config.BACKBONE,
        pretrained=config.PRETRAINED,
        in_channels=config.MODEL_INPUT_CHANNELS,
        num_classes=1,
    )
    model = model.to(device)

    # 4. Loss, Optimizer, Scheduler
    criterion = loss.HybridLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 5. Training Loop
    best_dice = 0.0
    best_epoch = -1

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch}/{epochs} | Time: {epoch_duration:.2f}s | LR: {current_lr:.8f}"
        )
        print(f"  Train Loss: {train_loss:.10f}")
        print(f"  Val Loss:   {val_loss:.10f}")
        print(f"  Val Dice:   {val_dice:.10f}")

        # Save Best Model
        if val_dice > best_dice:
            print(f"  -> New Best Dice! (Previous: {best_dice:.10f})")
            best_dice = val_dice
            best_epoch = epoch
            save_path = os.path.join(config.OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model saved to {save_path}")

        print("-" * 30)

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s.")
    print(f"Best Global Dice: {best_dice:.10f} at Epoch {best_epoch}")
