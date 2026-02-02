import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import (
    set_seed,
    AverageMeter,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
)
from library.dataset import get_dataloaders
from library.model import build_model


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        acc1 = (outputs.argmax(dim=1) == targets).float().mean() * 100.0
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

    print(f"Epoch: [{epoch}] Train Loss: {losses.avg:.6f} Acc: {top1.avg:.2f}%")
    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            acc1 = (outputs.argmax(dim=1) == targets).float().mean() * 100.0
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

    # Print full precision as requested
    print(f"Validation Loss: {losses.avg}")
    print(f"Validation Acc: {top1.avg:.2f}%")
    return losses.avg


def predict_tta(model, loader, device, classes):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).
    """
    model.eval()
    results = []
    ids_list = []

    print("Starting inference with TTA (Original + Horizontal Flip)...")

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # 1. Original Image
            outputs_orig = model(images)
            probs_orig = F.softmax(outputs_orig, dim=1)

            # 2. Flipped Image (Horizontal Flip)
            # NCHW format, width is dim 3
            images_flipped = torch.flip(images, dims=[3])
            outputs_flipped = model(images_flipped)
            probs_flipped = F.softmax(outputs_flipped, dim=1)

            # Average probabilities
            avg_probs = (probs_orig + probs_flipped) / 2.0

            results.append(avg_probs.cpu().numpy())
            ids_list.extend(ids)

    # Concatenate all batch results
    final_probs = np.concatenate(results, axis=0)

    # Create DataFrame
    df_sub = pd.DataFrame(final_probs, columns=classes)
    df_sub.insert(0, "id", ids_list)

    return df_sub


def main():
    # 1. Setup
    set_seed(Config.seed)
    device = torch.device(Config.device)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        load_cached_data=True
    )
    print(f"Classes: {len(classes)}")

    # 3. Model Construction
    model = build_model(num_classes=Config.num_classes, pretrained=Config.pretrained)
    model = model.to(device)

    # 4. Training Configuration
    criterion = nn.CrossEntropyLoss()

    # --- Phase 1: Warm-up ---
    print("\n" + "=" * 30)
    print("PHASE 1: Warm-up (Head Only)")
    print("=" * 30)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze head (timm specific)
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    optimizer_warmup = optim.AdamW(
        model.get_classifier().parameters(), lr=Config.lr_warmup
    )

    for epoch in range(1, Config.warmup_epochs + 1):
        train_one_epoch(model, train_loader, optimizer_warmup, criterion, device, epoch)
        validate(model, val_loader, criterion, device)

    # --- Phase 2: Fine-tuning ---
    print("\n" + "=" * 30)
    print("PHASE 2: Fine-tuning (Full Model)")
    print("=" * 30)

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr_fine_tune, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.scheduler_t_max
    )

    early_stopping = EarlyStopping(
        patience=7, verbose=True, path=Config.best_model_path
    )

    for epoch in range(1, Config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step()

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("\n" + "=" * 30)
    print("INFERENCE")
    print("=" * 30)

    # Load best model weights
    print(f"Loading best model from {Config.best_model_path}...")
    load_checkpoint(model, Config.best_model_path)

    # Generate predictions
    df_submission = predict_tta(model, test_loader, device, classes)

    # Save submission
    print(f"Saving submission to {Config.submission_path}...")
    df_submission.to_csv(Config.submission_path, index=False)

    print("Training and Inference pipeline completed successfully.")
