import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.utils import AverageMeter, save_checkpoint
from library.config import Config


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using BCEWithLogitsLoss.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        # Ensure targets are float and have shape (N, 1) for BCEWithLogitsLoss
        targets = targets.to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set and returns the average Log Loss.
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1).float()

            outputs = model(images)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), images.size(0))

    return losses.avg


def fit(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    learning_rate,
    weight_decay,
    save_dir,
    patience=3,
):
    """
    Runs the training pipeline with AdamW optimizer, Cosine Annealing scheduler, and Early Stopping.
    """
    # Optimizer: AdamW
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    patience_counter = 0

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate(model, val_loader, device)

        # Update scheduler at the end of the epoch
        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Save Checkpoint & Early Stopping Logic
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
            },
            is_best,
            save_dir,
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break


def predict(model, loader, device, use_tta=False):
    """
    Generates predictions for the test set.
    Supports Test Time Augmentation (Horizontal Flip).

    Returns:
        ids (list): List of image IDs.
        probs (list): List of predicted probabilities (0-1).
    """
    model.eval()
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # Standard forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            if use_tta:
                # Horizontal Flip TTA
                # Flip along width dimension (dim 3 for NCHW)
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped)

                # Average probabilities
                probs = (probs + probs_flipped) / 2.0

            all_probs.extend(probs.cpu().numpy().flatten())
            all_ids.extend(ids.numpy().flatten())

    return all_ids, all_probs


def generate_submission(ids, probs, output_path):
    """
    Saves the predictions to a CSV file in the required format.
    """
    df = pd.DataFrame({"id": ids, "label": probs})

    # Sort by ID to ensure consistency
    df = df.sort_values("id")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
