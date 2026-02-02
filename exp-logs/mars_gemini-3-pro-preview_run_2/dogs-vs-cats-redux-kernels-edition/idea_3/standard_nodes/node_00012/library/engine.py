import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library import config
from library import utils


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Applies Mixup augmentation to inputs and targets.
    Returns mixed inputs and mixed targets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    # Mix targets (works for binary float labels 0.0/1.0)
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y


def train_one_epoch(model, loader, optimizer, device, epoch, mixup_alpha):
    """
    Trains the model for one epoch using Mixup.
    """
    model.train()
    losses = utils.AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup
        images, labels = mixup_data(images, labels, alpha=mixup_alpha, device=device)

        # Forward pass
        # Model outputs shape [batch_size, 1] or [batch_size] depending on timm version/config
        # We ensure it's flattened to match labels
        outputs = model(images).view(-1)

        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns Log Loss and Accuracy.
    """
    model.eval()
    losses = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            preds = torch.sigmoid(outputs)
            # Threshold at 0.5
            predicted_labels = (preds > 0.5).float()
            acc = (predicted_labels == labels).float().mean()

            losses.update(loss.item(), images.size(0))
            acc_meter.update(acc.item(), images.size(0))

    return losses.avg, acc_meter.avg


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=config.EPOCHS,
    mixup_alpha=config.MIXUP_ALPHA,
    patience=3,
):
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    best_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, mixup_alpha
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device)

        # Scheduler Step
        if scheduler:
            scheduler.step()

        duration = time.time() - start_time

        # Print metrics with full precision
        print(f"Epoch {epoch+1}/{epochs} | Time: {duration:.2f}s")
        print(f"  Train Loss: {train_loss}")
        print(f"  Val Loss:   {val_loss}")
        print(f"  Val Acc:    {val_acc}")

        # Checkpointing
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
            )
            print("  New best model saved!")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_loss


def predict_and_submit(model, test_loader, device, output_path=config.SUBMISSION_PATH):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip)
    and saves them to a CSV file.
    """
    print("Generating predictions with TTA...")
    model.eval()

    results = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # 1. Forward pass original
            logits_orig = model(images).view(-1)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Forward pass flipped (TTA)
            images_flipped = torch.flip(
                images, dims=[3]
            )  # Flip width dimension (B, C, H, W)
            logits_flip = model(images_flipped).view(-1)
            probs_flip = torch.sigmoid(logits_flip)

            # Average probabilities
            probs_avg = (probs_orig + probs_flip) / 2.0

            # Store results
            ids_np = ids.cpu().numpy()
            probs_np = probs_avg.cpu().numpy()

            for img_id, prob in zip(ids_np, probs_np):
                results.append({"id": img_id, "label": prob})

    # Create DataFrame and save
    submission_df = pd.DataFrame(results)
    # Ensure ID is int
    submission_df["id"] = submission_df["id"].astype(int)
    # Sort by ID just in case
    submission_df = submission_df.sort_values("id")

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission_df.head())
