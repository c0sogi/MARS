import os
import time
import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.utils import MetricMonitor, calculate_roc_auc
from library.model import ModelEMA


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes loss for mixed inputs.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(
    model, train_loader, optimizer, scheduler, device, epoch, ema_model=None
):
    """
    Trains the model for one epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    for batch in train_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float()

        batch_size = images.size(0)

        # Apply Mixup
        if np.random.rand() < Config.MIXUP_PROB:
            mixed_images, y_a, y_b, lam = mixup_data(
                images, labels, Config.MIXUP_ALPHA, device
            )
            outputs = model(mixed_images).squeeze(1)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA model
        if ema_model is not None:
            ema_model.update(model)

        metric_monitor.update("Loss", loss.item(), batch_size)

    # Step scheduler at the end of the epoch
    if scheduler is not None:
        scheduler.step()

    return metric_monitor.avg["Loss"]


def validate(model, val_loader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC.
    """
    model.eval()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    preds = []
    targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            metric_monitor.update("Loss", loss.item(), images.size(0))

            # Store predictions (sigmoid) and targets for AUC calculation
            preds.append(torch.sigmoid(outputs).cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    auc = calculate_roc_auc(targets, preds)

    return metric_monitor.avg["Loss"], auc


def inference_fn(model, test_loader, device):
    """
    Generates predictions for the test set using TTA (8 views).
    Returns a numpy array of probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)

            # TTA: 8 views (Dihedral Group D4)
            # 1. Original
            # 2-4. Rotations 90, 180, 270
            # 5. Horizontal Flip
            # 6-8. Rotations of the flipped image

            batch_preds = []

            # Standard views
            for k in [0, 1, 2, 3]:
                img_rot = torch.rot90(images, k, [2, 3])
                out = model(img_rot).squeeze(1)
                batch_preds.append(torch.sigmoid(out))

            # Flipped views
            images_flipped = torch.flip(images, [3])  # Flip width/horizontal
            for k in [0, 1, 2, 3]:
                img_rot = torch.rot90(images_flipped, k, [2, 3])
                out = model(img_rot).squeeze(1)
                batch_preds.append(torch.sigmoid(out))

            # Stack and average
            batch_preds = torch.stack(batch_preds, dim=0)  # (8, B)
            avg_preds = torch.mean(batch_preds, dim=0)  # (B,)

            preds.append(avg_preds.cpu().numpy())

    return np.concatenate(preds)


def train_fold(
    fold_idx, train_loader, val_loader, model, optimizer, scheduler, device, patience=5
):
    """
    Orchestrates training for a single fold, including Early Stopping.
    """
    print(f"\n===== Starting Training for Fold {fold_idx} =====")

    best_auc = 0.0
    best_epoch = 0
    early_stopping_counter = 0

    # Initialize EMA
    ema = ModelEMA(model) if Config.USE_EMA else None

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, ema_model=ema
        )

        # Validate (use EMA model if available)
        val_model = ema.ema_model if ema else model
        val_loss, val_auc = validate(val_model, val_loader, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.8f}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            early_stopping_counter = 0

            save_path = os.path.join(
                Config.CHECKPOINT_DIR, f"best_model_fold_{fold_idx}.pth"
            )
            torch.save(val_model.state_dict(), save_path)
            print(f"  >>> Model Saved (Best AUC: {best_auc:.8f})")
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch}. Best AUC: {best_auc:.8f} at epoch {best_epoch}."
            )
            break

    return best_auc
