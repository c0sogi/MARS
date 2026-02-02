import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter


def train_one_epoch(
    epoch,
    model,
    train_loader,
    optimizer,
    device,
    loss_fn,
    mixup_fn=None,
    model_ema=None,
):
    """
    Trains the model for one epoch.
    Handles MixUp/CutMix application, gradient clipping, and EMA updates.
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for step, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply MixUp / CutMix if enabled
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        if Config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        # Update EMA model
        if model_ema is not None:
            model_ema.update(model)

        # Update Loss Meter
        loss_meter.update(loss.item(), images.size(0))

        # Calculate Accuracy
        # Handle soft targets (MixUp) vs hard targets (Indices)
        if targets.ndim == 2:
            # Soft targets: argmax for accuracy
            preds = outputs.argmax(dim=1)
            targs = targets.argmax(dim=1)
        else:
            # Hard targets
            preds = outputs.argmax(dim=1)
            targs = targets

        acc = (preds == targs).float().mean().item()
        acc_meter.update(acc, images.size(0))

    # Print metrics with full precision
    print(f"Epoch {epoch} | Train Loss: {loss_meter.avg} | Train Acc: {acc_meter.avg}")

    return loss_meter.avg, acc_meter.avg


def validate(model, val_loader, device, loss_fn):
    """
    Evaluates the model on the validation set.
    Returns average loss and accuracy.
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, targets)

            # Update Loss Meter
            loss_meter.update(loss.item(), images.size(0))

            # Calculate Accuracy
            preds = outputs.argmax(dim=1)
            acc = (preds == targets).float().mean().item()
            acc_meter.update(acc, images.size(0))

    # Print metrics with full precision
    print(f"Validation | Loss: {loss_meter.avg} | Acc: {acc_meter.avg}")

    return loss_meter.avg, acc_meter.avg


def inference(models, test_loader, device):
    """
    Generates predictions for the test set using an ensemble of models.
    Supports Test Time Augmentation (TTA) by averaging predictions of original and flipped images.
    Saves predictions to submission.csv.
    """
    # Ensure all models are in eval mode
    for model in models:
        model.eval()

    all_preds = []

    # Disable gradient calculation
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)
            batch_size = images.size(0)

            # Initialize aggregated probabilities
            avg_probs = torch.zeros(batch_size, Config.NUM_CLASSES).to(device)

            for model in models:
                # 1. Standard View
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                # 2. TTA: Horizontal Flip
                if Config.USE_TTA:
                    # Flip width dimension (N, C, H, W) -> dim 3
                    images_flipped = torch.flip(images, dims=[3])
                    outputs_flipped = model(images_flipped)
                    probs_flipped = torch.softmax(outputs_flipped, dim=1)

                    # Average original and flipped probabilities
                    probs = (probs + probs_flipped) / 2.0

                # Accumulate
                avg_probs += probs

            # Average over all models in the ensemble
            avg_probs /= len(models)

            # Get final predictions
            preds = avg_probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    # Create Submission DataFrame
    # Note: We assume test_loader.dataset is the CassavaDataset which holds the dataframe
    test_df = test_loader.dataset.df

    submission = pd.DataFrame({"image_id": test_df["image_id"], "label": all_preds})

    # Save submission
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
