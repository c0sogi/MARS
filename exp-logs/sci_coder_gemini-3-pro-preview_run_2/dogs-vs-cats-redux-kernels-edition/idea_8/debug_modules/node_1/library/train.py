import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, AverageMeter, get_score
from library.dataset import get_train_val_loaders
from library.models import get_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)
        batch_size = images.size(0)

        # Mixup Augmentation
        if Config.MIXUP_ALPHA > 0:
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index]
            y_a, y_b = labels, labels[index]

            logits = model(mixed_images)
            # Mix the loss
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def validate_one_epoch(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, labels)

            losses.update(loss.item(), batch_size)

            # Apply sigmoid to get probabilities for scoring
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    score = get_score(targets, preds)

    return losses.avg, score


def run_fold(model_name, fold_idx):
    """
    Runs the training loop for a specific model architecture and fold.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    # We use cached data if available to speed up multi-process runs
    train_loader, val_loader = get_train_val_loaders(fold_idx, load_cached_data=True)

    # Initialize Model
    model = get_model(model_name, pretrained=True)
    model.to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_score = float("inf")

    print(f"Starting training for {model_name} - Fold {fold_idx}")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics with full precision as requested
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss} | "
            f"Val Score: {val_score} | "
            f"Time: {elapsed:.2f}s"
        )

        # Save Checkpoint (Best Loss)
        # We save based on Loss as it is the primary optimization objective
        if val_loss < best_loss:
            best_loss = val_loss
            best_score = val_score

            save_name = f"{model_name}_fold_{fold_idx}.pth"
            save_path = os.path.join(Config.CHECKPOINT_DIR, save_name)
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print(
        f"Fold {fold_idx} finished. Best Val Loss: {best_loss}, Best Val Score: {best_score}"
    )

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_loss
