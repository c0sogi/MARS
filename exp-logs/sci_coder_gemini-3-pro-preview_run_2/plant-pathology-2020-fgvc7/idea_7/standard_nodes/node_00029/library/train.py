import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, calculate_metric, save_model
from library.data import get_loaders
from library.model import AppleClassifier


class SmoothBCE(nn.Module):
    """
    Binary Cross Entropy Loss with Logits, supporting:
    1. Positive Class Weights (for imbalance)
    2. Label Smoothing (for regularization)
    """

    def __init__(self, pos_weight=None, smoothing=0.0):
        super(SmoothBCE, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        if self.smoothing > 0:
            # Apply label smoothing: y_new = y * (1 - alpha) + 0.5 * alpha
            # This assumes binary targets 0 or 1
            with torch.no_grad():
                targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


def train_one_epoch(epoch, model, loader, optimizer, criterion, device, scheduler=None):
    """
    Trains the model for one epoch.
    Handles Multi-Sample Dropout (MSD) by averaging loss across multiple heads.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        # In training mode, model returns a list of logits (MSD)
        outputs = model(images)

        # Calculate loss
        # Average the loss across all dropout masks
        loss = 0
        for output in outputs:
            loss += criterion(output, targets)
        loss /= len(outputs)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    return avg_loss


def validate_one_epoch(epoch, model, loader, criterion, device):
    """
    Validates the model.
    Computes Loss and ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            # Forward pass
            # In eval mode, model returns a single tensor of logits (averaged internally)
            logits = model(images)

            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Metric (Mean Column-wise ROC AUC)
    auc_score = calculate_metric(all_targets, all_preds)

    return avg_loss, auc_score


def run_fold(fold_idx, model_name, img_size):
    """
    Executes the training pipeline for a single fold.

    Args:
        fold_idx (int): Index of the current fold (0-4).
        model_name (str): Name of the timm backbone to use.
        img_size (int): Resolution for image resizing.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting Fold {fold_idx} | Model: {model_name} | Size: {img_size}")

    # 1. Data Loaders
    train_loader, val_loader, pos_weights = get_loaders(fold_idx, img_size)

    # Move class weights to device for Loss function
    pos_weights = pos_weights.to(device)

    # 2. Model
    model = AppleClassifier(model_name=model_name, pretrained=True)
    model.to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # 4. Loss Function
    criterion = SmoothBCE(pos_weight=pos_weights, smoothing=Config.LABEL_SMOOTHING)

    # 5. Training Loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    # Sanitize model name for filename
    safe_model_name = model_name.replace(".", "_")
    save_path = os.path.join(
        Config.WORKING_DIR, f"best_model_{safe_model_name}_fold_{fold_idx}.pth"
    )

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc = validate_one_epoch(
            epoch, model, val_loader, criterion, device
        )

        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} [{elapsed:.0f}s] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc}"
        )

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            save_model(model, save_path)
            print(f"  --> Saved Best Model (AUC: {best_auc})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Fold {fold_idx} finished. Best AUC: {best_auc}")

    # Clear memory
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_auc
