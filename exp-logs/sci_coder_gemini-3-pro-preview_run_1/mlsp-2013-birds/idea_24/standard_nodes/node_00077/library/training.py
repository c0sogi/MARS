import os
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
import numpy as np

from library.config import Config
from library.utils import AverageMeter, get_logger
from library.dataset import Mixup


def train_one_epoch(model, loader, optimizer, criterion, device, mixup_fn):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if mixup_fn is not None:
            images, y_a, y_b, lam = mixup_fn(images, targets)
            outputs = model(images)
            loss = criterion(outputs, y_a) * lam + criterion(outputs, y_b) * (1 - lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    if len(all_preds) == 0:
        return 0.0, 0.0

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Macro-Averaged ROC AUC
    # Handle cases where a class might not be present in the validation set
    # Cite debug_lesson_5: Safeguard Global Metrics Against Degenerate Data Subsets
    aucs = []
    for i in range(all_targets.shape[1]):
        try:
            # Only calculate AUC if both classes (0 and 1) are present
            if len(np.unique(all_targets[:, i])) > 1:
                aucs.append(roc_auc_score(all_targets[:, i], all_preds[:, i]))
            else:
                # Fallback for degenerate classes
                aucs.append(0.5)
        except ValueError:
            aucs.append(0.5)

    auc = np.mean(aucs) if aucs else 0.5

    return losses.avg, auc


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    mixup_alpha,
    swa_start_epoch,
    swa_lr,
    epochs,
    save_dir,
    model_alias,
    patience=10,
):
    """
    Main training loop with SWA and Early Stopping.
    """
    # Setup logging
    log_path = os.path.join(save_dir, f"{model_alias}_training.log")
    logger = get_logger(log_path)
    logger.info(f"Starting training for {model_alias}")
    logger.info(
        f"Mixup Alpha: {mixup_alpha}, SWA Start: {swa_start_epoch}, SWA LR: {swa_lr}"
    )

    criterion = nn.BCEWithLogitsLoss()

    # Scheduler for the base training phase
    # Decays LR from initial value to swa_lr by the time SWA starts
    scheduler = CosineAnnealingLR(optimizer, T_max=swa_start_epoch, eta_min=swa_lr)

    # SWA Setup
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

    # Mixup Setup
    mixup_fn = Mixup(alpha=mixup_alpha) if mixup_alpha > 0 else None

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        current_epoch = epoch + 1
        is_swa_phase = epoch >= swa_start_epoch

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, mixup_fn
        )

        # Update SWA or Standard Scheduler
        if is_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            lr = swa_scheduler.get_last_lr()[0]
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {current_epoch}/{epochs} | "
            f"LR: {lr:.8f} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Save Best Base Model and Handle Early Stopping
        # We only apply Early Stopping BEFORE SWA starts.
        # Once SWA starts, we continue until the end to collect averages.
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save best base model
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"{model_alias}_base_best.pth"),
            )
        else:
            patience_counter += 1

        if not is_swa_phase and patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {current_epoch}")
            break

    # Save Last Base Model
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_alias}_last.pth"))

    # Finalize SWA
    if epochs > swa_start_epoch:
        logger.info("Updating SWA BatchNorm statistics...")
        # update_bn expects the loader to yield samples.
        # BirdDataset yields (image, label), update_bn handles unpacking.
        update_bn(train_loader, swa_model, device=device)

        # Evaluate SWA Model
        swa_val_loss, swa_val_auc = validate(swa_model, val_loader, criterion, device)
        logger.info(
            f"SWA Final Results | Val Loss: {swa_val_loss:.8f} | Val AUC: {swa_val_auc:.10f}"
        )

        # Save SWA Model
        torch.save(
            swa_model.state_dict(), os.path.join(save_dir, f"{model_alias}_swa.pth")
        )
        return swa_model
    else:
        logger.info("SWA phase was not reached. Returning best base model.")
        # Load best weights if SWA wasn't reached
        model.load_state_dict(
            torch.load(os.path.join(save_dir, f"{model_alias}_base_best.pth"))
        )
        return model
