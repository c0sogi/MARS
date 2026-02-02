import os
import time
import numpy as np
import torch
import torch.nn as nn
from library.utils import (
    AverageMeter,
    calculate_roc_auc,
    save_checkpoint,
    load_checkpoint,
)


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
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
    Calculates the mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(train_loader, model, criterion, optimizer, device, config):
    """
    Trains the model for one epoch using Mixup.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        # BCEWithLogitsLoss expects targets of shape (N, 1) usually
        targets = targets.to(device).unsqueeze(1)

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, targets, config.mixup_alpha, device
        )

        # Forward pass
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC.
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    auc = calculate_roc_auc(all_targets, all_preds)
    return losses.avg, auc


def predict(loader, model, device, tta_steps=1):
    """
    Generates predictions for a dataloader, optionally using Test-Time Augmentation (TTA).

    TTA Strategies:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    """
    model.eval()
    accumulated_preds = []

    # 1. Original Pass
    preds_1 = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            preds_1.append(torch.sigmoid(outputs).cpu())
    accumulated_preds.append(torch.cat(preds_1, dim=0))

    # 2. Horizontal Flip Pass
    if tta_steps >= 2:
        preds_2 = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                images = torch.flip(images, [3])  # Flip W dimension
                outputs = model(images)
                preds_2.append(torch.sigmoid(outputs).cpu())
        accumulated_preds.append(torch.cat(preds_2, dim=0))

    # 3. Vertical Flip Pass
    if tta_steps >= 3:
        preds_3 = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                images = torch.flip(images, [2])  # Flip H dimension
                outputs = model(images)
                preds_3.append(torch.sigmoid(outputs).cpu())
        accumulated_preds.append(torch.cat(preds_3, dim=0))

    # Average predictions across TTA steps
    # Stack -> (TTA, N, 1) -> Mean -> (N, 1)
    avg_preds = torch.stack(accumulated_preds).mean(dim=0)

    return avg_preds.numpy()


def fit_model(
    config, model, train_loader, val_loader, fold_id, save_name="best_model.pth"
):
    """
    Main training loop for a single model on a single fold.
    Handles training, validation, early stopping, and OOF prediction.
    """
    device = config.device
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # Cosine Annealing Scheduler (Cite solution_lesson_node_00016)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    save_path = os.path.join(config.output_dir, save_name)

    print(f"Starting training for Fold {fold_id}...")

    for epoch in range(config.epochs):
        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, config
        )

        # Step scheduler
        scheduler.step()

        # Validate
        val_loss, val_auc = validate(val_loader, model, criterion, device)

        # Print metrics (full precision)
        print(
            f"Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(model, save_path)
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Fold {fold_id} finished. Best AUC: {best_auc} at epoch {best_epoch+1}")

    # Load best model for OOF predictions
    model = load_checkpoint(model, save_path, device=device)

    # Generate OOF predictions using TTA
    oof_preds = predict(val_loader, model, device, tta_steps=config.tta_steps)

    # Extract OOF targets
    oof_targets = []
    for _, targets in val_loader:
        oof_targets.append(targets.numpy())
    oof_targets = np.concatenate(oof_targets)

    return oof_preds, oof_targets, best_auc
