import os
import time
import torch
import torch.nn as nn
import numpy as np
from library.config import WORKING_DIR, DEVICE, ModelConfig
from library.utils import compute_log_loss, save_checkpoint
from library.data import get_fold_loaders
from library.models import create_model


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape [batch, 1]

        optimizer.zero_grad()

        # Forward pass (outputs are logits)
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss (BCE) and the calculated Log Loss metric.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Apply sigmoid for probability calculation
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size

    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    # Compute metric
    metric_log_loss = compute_log_loss(all_labels, all_preds)

    return avg_loss, metric_log_loss


def run_fold(fold_idx: int, cfg: ModelConfig):
    """
    Runs the training and validation loop for a specific fold and model configuration.
    Saves the best model checkpoint based on validation loss.
    """
    print(f"--- Starting Fold {fold_idx} for {cfg.model_name} ---")

    # 1. Data Loaders
    train_loader, val_loader = get_fold_loaders(fold_idx, cfg)

    # 2. Model Setup
    model = create_model(cfg, pretrained=True)
    model = model.to(DEVICE)

    # 3. Loss, Optimizer, Scheduler
    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
    )

    best_val_loss = float("inf")
    best_epoch = -1

    # Checkpoint filename
    checkpoint_name = f"{cfg.model_name}_fold_{fold_idx}.pth"

    # 4. Training Loop
    for epoch in range(cfg.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, epoch
        )
        val_loss, val_metric = evaluate(model, val_loader, criterion, DEVICE)

        # Step the scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric: {val_metric} | "
            f"Time: {elapsed:.2f}s"
        )

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": cfg,
                },
                filename=checkpoint_name,
            )

            print(f"  >>> Model saved (Improved loss: {best_val_loss})")

    print(
        f"Fold {fold_idx} completed. Best Val Loss: {best_val_loss} at Epoch {best_epoch}"
    )
    return best_val_loss
