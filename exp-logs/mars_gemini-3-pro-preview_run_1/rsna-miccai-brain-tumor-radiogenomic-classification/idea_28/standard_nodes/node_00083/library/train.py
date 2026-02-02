import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import (
    seed_everything,
    get_device,
    calculate_roc_auc,
    print_metric,
    AverageMeter,
)
from library.model import MGMTModel
from library.data import get_dataloader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # (Batch, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate_one_epoch(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    losses = AverageMeter()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets_gpu = targets.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, targets_gpu)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(logits)

            all_targets.append(targets.numpy())
            all_preds.append(preds.cpu().numpy())

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    auc_score = calculate_roc_auc(all_targets, all_preds)

    return losses.avg, auc_score


def run_training_fold(fold_idx):
    """
    Runs the training pipeline for a specific fold.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED + fold_idx)
    device = get_device()

    print(f"\n{'='*20} Starting Fold {fold_idx} {'='*20}")

    # 1. Data Loaders
    train_loader = get_dataloader(
        split="train",
        fold_idx=fold_idx,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = get_dataloader(
        split="val",
        fold_idx=fold_idx,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Model Initialization
    model = MGMTModel(
        model_name=Config.MODEL_NAME,
        pretrained=True,
        num_classes=1,
        dropout_rate=Config.DROPOUT_RATE,
    )
    model = model.to(device)

    # 3. Optimization Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    # 4. Training Loop with Early Stopping
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    # Ensure cache directory exists for saving models
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    save_path = os.path.join(Config.CACHE_DIR, f"best_model_fold{fold_idx}.pth")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc}"
        )  # Printing full precision as requested

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New Best AUC! Model saved to {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs with no improvement."
            )
            break

    print(f"Fold {fold_idx} finished. Best Validation AUC: {best_auc}")
    return best_auc


def train_models():
    """
    Main entry point to train all folds.
    """
    fold_scores = []

    for fold in range(Config.N_FOLDS):
        score = run_training_fold(fold)
        fold_scores.append(score)

    avg_score = np.mean(fold_scores)
    print("\n" + "=" * 40)
    print(f"Cross-Validation Complete.")
    print(f"Fold Scores: {fold_scores}")
    print(f"Average AUC: {avg_score}")
    print("=" * 40)
