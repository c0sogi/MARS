import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from library.model import DepthConditionedUNetPlusPlus
from library.dataset import SaltDataset, get_transforms
from library.losses import BCEDiceLoss
from library.utils import MetricMonitor, set_seed

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./working/idea_2/"
os.makedirs(SAVE_DIR, exist_ok=True)


def calculate_iou(outputs, targets, threshold=0.5):
    """
    Calculates IoU for a batch.
    Handles Deep Supervision outputs (list of tensors) by averaging probabilities.
    """
    with torch.no_grad():
        if isinstance(outputs, (list, tuple)):
            # Deep Supervision Ensemble: Average sigmoid probabilities
            preds = torch.zeros_like(outputs[0])
            for out in outputs:
                preds += torch.sigmoid(out)
            preds /= len(outputs)
        else:
            preds = torch.sigmoid(outputs)

        preds = (preds > threshold).float()

        # Ensure targets match preds shape (N, 1, H, W)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection

        iou = (intersection + 1e-7) / (union + 1e-7)
        return iou.item()


def train_one_epoch(model, train_loader, criterion, optimizer, metric_monitor):
    model.train()
    metric_monitor.reset()

    for batch_idx, (images, masks, depths, _) in enumerate(train_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        depths = depths.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, depths)

        # Loss calculation
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        iou = calculate_iou(outputs, masks)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("IoU", iou)


def validate(model, val_loader, criterion, metric_monitor):
    model.eval()
    metric_monitor.reset()

    with torch.no_grad():
        for batch_idx, (images, masks, depths, _) in enumerate(val_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            depths = depths.to(DEVICE)

            # Forward pass
            outputs = model(images, depths)

            # Loss calculation
            loss = criterion(outputs, masks)

            # Metrics
            iou = calculate_iou(outputs, masks)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("IoU", iou)


def run_training(
    epochs=50, batch_size=32, lr=1e-3, num_workers=4, patience=10, load_cached_data=True
):
    set_seed(42)

    print(f"Starting training on device: {DEVICE}")
    print(f"Hyperparameters: Epochs={epochs}, Batch Size={batch_size}, LR={lr}")

    # --- Data Loading ---
    train_dataset = SaltDataset(
        mode="train",
        metadata_path="./metadata/train.csv",
        load_cached_data=load_cached_data,
        transform=get_transforms("train"),
    )

    val_dataset = SaltDataset(
        mode="val",
        metadata_path="./metadata/val.csv",
        load_cached_data=load_cached_data,
        transform=get_transforms("val"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # --- Model Setup ---
    model = DepthConditionedUNetPlusPlus(num_classes=1, deep_supervision=True)
    model = model.to(DEVICE)

    # --- Optimizer, Scheduler, Loss ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    # --- Training Loop ---
    best_iou = -float("inf")
    patience_counter = 0
    train_monitor = MetricMonitor()
    val_monitor = MetricMonitor()

    model_save_path = os.path.join(SAVE_DIR, "best_model.pth")

    for epoch in range(1, epochs + 1):
        # Train
        train_one_epoch(model, train_loader, criterion, optimizer, train_monitor)

        # Validate
        validate(model, val_loader, criterion, val_monitor)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train | {train_monitor}")
        print(f"Val   | {val_monitor}")

        # Checkpointing & Early Stopping
        current_iou = val_monitor.metrics["IoU"]["avg"]

        if current_iou > best_iou:
            print(
                f"Validation IoU improved from {best_iou} to {current_iou}. Saving model..."
            )
            best_iou = current_iou
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation IoU: {best_iou}")
    print(f"Best model saved to: {model_save_path}")
