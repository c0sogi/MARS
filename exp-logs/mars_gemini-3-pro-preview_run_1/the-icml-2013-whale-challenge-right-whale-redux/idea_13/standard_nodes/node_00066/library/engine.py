import os
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, compute_auc, print_metric


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns:
        mixed_x: Mixed inputs
        y_a: Targets for the first image
        y_b: Targets for the second image
        lam: Lambda mixing coefficient
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
    Computes the Mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        # Ensure target is (N, 1) for BCEWithLogitsLoss
        target = target.to(device).view(-1, 1)

        optimizer.zero_grad()

        if Config.DO_MIXUP and Config.MIXUP_ALPHA > 0:
            data, target_a, target_b, lam = mixup_data(
                data, target, Config.MIXUP_ALPHA, device
            )
            output = model(data)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            output = model(data)
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), data.size(0))

    return losses.avg


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns:
        avg_loss (float): Average validation loss.
        auc (float): Area Under the ROC Curve.
    """
    model.eval()
    losses = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device).view(-1, 1)

            output = model(data)
            loss = criterion(output, target)

            losses.update(loss.item(), data.size(0))

            # Apply sigmoid to logits to get probabilities
            preds = torch.sigmoid(output)

            all_targets.append(target.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Compute AUC
    auc = compute_auc(all_targets, all_preds)

    return losses.avg, auc


def train_model(
    model, train_loader, val_loader, optimizer, scheduler, device, epochs=Config.EPOCHS
):
    """
    Orchestrates the training process with Early Stopping and Checkpointing.
    """
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    # Initialize Loss with Positive Class Weight
    pos_weight = Config.get_pos_weight_tensor().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # Training Step
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validation Step
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Scheduler Step (Monitor AUC)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        # Logging
        print(f"Epoch {epoch}/{epochs}")
        print_metric("Train Loss", train_loss)
        print_metric("Val Loss", val_loss)
        print_metric("Val AUC", val_auc)

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {best_auc}")
    return best_auc
