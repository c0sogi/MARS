import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import seed_everything, compute_kappa_score, ModelEMA
from library.dataset import create_dataloaders
from library.model import RetinaModel


def get_ordinal_targets(labels, num_classes):
    """
    Converts integer labels to ordinal binary targets.
    Args:
        labels (torch.Tensor): Integer labels (0 to 4).
        num_classes (int): Number of binary units (4).
    Returns:
        torch.Tensor: Binary targets of shape [batch_size, num_classes].
    """
    batch_size = labels.size(0)
    targets = torch.zeros(batch_size, num_classes, device=labels.device)
    for i in range(num_classes):
        # Target is 1 if the label is greater than the current rank index
        # i.e., for label 2: >0 is True, >1 is True, >2 is False, >3 is False -> [1, 1, 0, 0]
        targets[:, i] = (labels > i).float()
    return targets


def train_one_epoch(model, loader, criterion, optimizer, scaler, ema, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        with autocast(enabled=Config.use_amp):
            outputs = model(images)
            targets = get_ordinal_targets(labels, Config.num_classes)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema:
            ema.update(model)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)

            # Loss calculation
            targets = get_ordinal_targets(labels, Config.num_classes)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Decoding: Sum probabilities and round
            probs = torch.sigmoid(outputs)
            preds = probs.sum(dim=1).round()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    kappa = compute_kappa_score(all_labels, all_preds)

    return epoch_loss, kappa


def inference_tta(model, loader, device):
    """
    Performs inference with 4-View Test Time Augmentation.
    """
    model.eval()
    all_preds = []
    all_ids = []  # Assuming loader preserves order, but we iterate sequentially

    # We need to get IDs. The dataset returns (image, label).
    # The test CSV has 'id_code'. We will rely on the loader order matching the CSV.

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 4 Views: Original, HFlip, VFlip, Rotate180
            # Rotate180 is equivalent to HFlip + VFlip

            # View 1: Original
            out1 = torch.sigmoid(model(images))

            # View 2: Horizontal Flip
            out2 = torch.sigmoid(model(torch.flip(images, [3])))

            # View 3: Vertical Flip
            out3 = torch.sigmoid(model(torch.flip(images, [2])))

            # View 4: Rotate 180 (HFlip + VFlip)
            out4 = torch.sigmoid(model(torch.flip(images, [2, 3])))

            # Average probabilities
            avg_probs = (out1 + out2 + out3 + out4) / 4.0

            # Decode: Sum and round
            preds = avg_probs.sum(dim=1).round()
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds).astype(int)


def run():
    """
    Main execution function.
    """
    seed_everything(Config.seed)

    # Create Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(load_cached_data=True)

    # Initialize Model
    model = RetinaModel()
    model.to(Config.device)

    # Initialize EMA
    ema = (
        ModelEMA(model, decay=Config.ema_decay, device=Config.device)
        if Config.use_ema
        else None
    )

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    # Loss Function (Summed across ordinal units)
    criterion = nn.BCEWithLogitsLoss()

    # Mixed Precision Scaler
    scaler = GradScaler(enabled=Config.use_amp)

    best_kappa = -float("inf")

    print(f"Starting training on device: {Config.device}")
    print(f"Model: {Config.model_name}, Image Size: {Config.image_size}")

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, ema, Config.device
        )

        # Validate (Use EMA model for validation if available)
        val_model = ema.module if ema else model
        val_loss, val_kappa = validate(val_model, val_loader, criterion, Config.device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Kappa: {val_kappa}"
        )

        # Save Best Model
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(val_model.state_dict(), Config.best_model_path)
            # print(f"New best model saved with Kappa: {best_kappa}")

        # Save Last Model
        torch.save(val_model.state_dict(), Config.last_model_path)

    print(f"Training complete. Best Validation Kappa: {best_kappa}")

    # ==========================================
    # Inference on Test Set
    # ==========================================
    print("Starting Inference on Test Set with TTA...")

    # Load Best Model
    best_model = RetinaModel()
    best_model.load_state_dict(
        torch.load(Config.best_model_path, map_location=Config.device)
    )
    best_model.to(Config.device)

    # Generate Predictions
    preds = inference_tta(best_model, test_loader, Config.device)

    # Create Submission DataFrame
    # Load test metadata to get ID codes
    test_df = pd.read_csv(Config.test_csv)

    submission = pd.DataFrame({"id_code": test_df["id_code"], "diagnosis": preds})

    # Save Submission
    submission.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
