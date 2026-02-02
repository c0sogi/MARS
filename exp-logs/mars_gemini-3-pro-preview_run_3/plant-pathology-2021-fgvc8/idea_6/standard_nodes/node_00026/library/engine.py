import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import MetricMonitor, calculate_f1_score


def train_one_epoch(
    model, train_loader, optimizer, device, epoch, model_ema=None, scaler=None
):
    """
    Trains the model for one epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()
    # BCEWithLogitsLoss combines Sigmoid and BCELoss, numerically stable and supports soft targets
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        with autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update EMA model
        if model_ema:
            model_ema.update(model)

        metric_monitor.update("Loss", loss.item())

    return metric_monitor.get_avg("Loss")


def validate(model, val_loader, device, threshold=0.5):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, targets)

            metric_monitor.update("Loss", loss.item())

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            preds_all.append(probs.float().cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    # Calculate F1 Score
    f1 = calculate_f1_score(
        targets_all, preds_all, threshold=threshold, average="macro"
    )

    return metric_monitor.get_avg("Loss"), f1


def predict_tta(model, test_loader, device, threshold=0.5):
    """
    Generates predictions for the test set using Test Time Augmentation (TTA).
    TTA: Original + Horizontal Flip + Vertical Flip.
    """
    model.eval()
    final_preds = []

    # Retrieve the dataframe from the dataset to access image IDs
    test_df = test_loader.dataset.df

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)

            with autocast(enabled=True):
                # 1. Forward pass on original images
                logits_orig = model(images)

                # 2. Forward pass on horizontally flipped images
                # dim 3 is width (B, C, H, W)
                images_hflip = torch.flip(images, dims=[3])
                logits_hflip = model(images_hflip)

                # 3. Forward pass on vertically flipped images
                # dim 2 is height
                images_vflip = torch.flip(images, dims=[2])
                logits_vflip = model(images_vflip)

                # Average logits
                avg_logits = (logits_orig + logits_hflip + logits_vflip) / 3.0

            # Convert to probabilities
            probs = torch.sigmoid(avg_logits)
            final_preds.append(probs.float().cpu().numpy())

    final_preds = np.concatenate(final_preds)

    # Process predictions into submission format
    submission_rows = []

    for idx, row in test_df.iterrows():
        image_id = row["image"]
        probs = final_preds[idx]

        # Get indices where probability > threshold
        label_indices = np.where(probs > threshold)[0]

        # Map indices to label names
        labels = [Config.ID2LABEL[i] for i in label_indices]

        # Join labels with space
        label_str = " ".join(labels)

        submission_rows.append({"image": image_id, "labels": label_str})

    return pd.DataFrame(submission_rows)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    checkpoint_dir,
    use_ema=False,
    model_ema=None,
    scaler=None,
):
    """
    Orchestrates the training loop, including validation and early stopping.
    """
    best_f1 = -1.0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # Train Step
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, model_ema, scaler
        )

        # Step Scheduler
        if scheduler:
            scheduler.step()

        # Validation Step
        # If EMA is used, we evaluate the EMA model as it usually generalizes better
        eval_model = model_ema.module if (use_ema and model_ema) else model
        val_loss, val_f1 = validate(
            eval_model, val_loader, device, threshold=Config.THRESHOLD
        )

        # Logging
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val F1: {val_f1:.6f}"
        )

        # Early Stopping / Checkpointing
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(eval_model.state_dict(), save_path)
            print(f"New Best F1 Score! Model saved to {save_path}")

    print(f"Training complete. Best Validation F1: {best_f1:.6f}")
