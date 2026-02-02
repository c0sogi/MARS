import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import get_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Metrics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        total += batch_size
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).
    Returns a numpy array of probabilities.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Forward pass on original images
            outputs_orig = model(images)
            probs_orig = F.softmax(outputs_orig, dim=1)

            # 2. Forward pass on flipped images (TTA)
            if Config.TTA_FLIP:
                # Flip along width dimension (N, C, H, W) -> dim 3
                images_flip = torch.flip(images, dims=[3])
                outputs_flip = model(images_flip)
                probs_flip = F.softmax(outputs_flip, dim=1)

                # Average probabilities
                probs = (probs_orig + probs_flip) / 2.0
            else:
                probs = probs_orig

            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def train_fold(fold_idx):
    """
    Orchestrates the training process for a single fold, including:
    - Warm-up phase
    - Fine-tuning phase
    - Saving checkpoints for Model Soup
    """
    # Ensure reproducibility for this fold
    seed_everything(Config.SEED + fold_idx)

    device = Config.DEVICE
    print(f"--- Starting Training for Fold {fold_idx} ---")

    # 1. Prepare Data
    train_loader, val_loader, _ = get_dataloaders(fold_idx=fold_idx)

    # 2. Prepare Model
    model = get_model(device=device, pretrained=True)
    criterion = nn.CrossEntropyLoss()

    # ==========================================
    # Phase 1: Warm-up (Head Only)
    # ==========================================
    print("Phase 1: Warm-up (Training Head Only)")
    model.freeze_backbone()

    # Optimizer for head parameters only
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        head_params, lr=Config.LR_WARMUP, weight_decay=Config.WEIGHT_DECAY
    )

    # Train for 1 epoch
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(
        f"Warmup Epoch - Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, "
        f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}"
    )

    # ==========================================
    # Phase 2: Fine-tuning (Full Model)
    # ==========================================
    print("Phase 2: Fine-tuning (Full Backbone)")
    model.unfreeze_backbone()

    # Re-initialize optimizer for all parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR_FINE_TUNE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX
    )

    soup_checkpoints = []

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Fold {fold_idx} Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr:.8f} | "
            f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}"
        )

        # Save checkpoints for Model Soup strategy
        if epoch >= Config.SOUP_EPOCH_START:
            ckpt_filename = f"fold_{fold_idx}_epoch_{epoch}.pth"
            ckpt_path = os.path.join(Config.OUTPUT_DIR, ckpt_filename)

            torch.save(model.state_dict(), ckpt_path)
            soup_checkpoints.append(ckpt_path)
            # print(f"Saved checkpoint for soup: {ckpt_filename}")

    return soup_checkpoints
