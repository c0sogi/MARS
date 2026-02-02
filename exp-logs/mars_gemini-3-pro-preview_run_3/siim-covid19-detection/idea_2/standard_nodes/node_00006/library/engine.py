import torch
import time
import os
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import get_one_stage_detector, train_one_epoch, validate_one_epoch
from library.dataset import ChestXRayDataset
from library.utils import collate_fn
from library.metrics import calculate_map


def evaluate(model, dataloader, device):
    """
    Runs inference on the validation set and computes mAP.

    Args:
        model: The PyTorch model.
        dataloader: Validation DataLoader.
        device: Torch device.

    Returns:
        dict: Dictionary containing 'map' and 'class_aps'.
    """
    model.eval()
    predictions = []
    targets_list = []

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)

            # In eval mode, the model returns a list of detections
            # [{'boxes': ..., 'scores': ..., 'labels': ...}, ...]
            preds = model(images)

            predictions.extend(preds)
            targets_list.extend(targets)

    # Calculate mAP
    metrics = calculate_map(predictions, targets_list)
    return metrics


def fit(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    load_cached_data=True,
    save_path=None,
    patience=5,
):
    """
    Main training loop with Early Stopping based on mAP.

    Args:
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay.
        load_cached_data (bool): Whether to load cached metadata.
        save_path (str): Path to save the best model.
        patience (int): Early stopping patience.
    """
    if save_path is None:
        save_path = Config.MODEL_SAVE_PATH

    print(f"Starting training on device: {Config.DEVICE}")

    # 1. Prepare Data
    train_dataset = ChestXRayDataset(split="train", load_cached_data=load_cached_data)
    val_dataset = ChestXRayDataset(split="val", load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 2. Prepare Model & Optimizer
    model = get_one_stage_detector()
    model.to(Config.DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Training Loop
    best_map = -1.0
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        # Train one epoch (returns average loss)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, Config.DEVICE, epoch
        )

        # Validate (returns average loss)
        val_loss = validate_one_epoch(model, val_loader, Config.DEVICE)

        # Evaluate (returns mAP metrics)
        val_metrics = evaluate(model, val_loader, Config.DEVICE)
        val_map = val_metrics["map"]

        scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics with full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val mAP: {val_map}"
        )

        # Checkpointing & Early Stopping based on mAP
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved! (mAP: {val_map})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best mAP: {best_map}")
