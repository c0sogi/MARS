import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import compute_auc, save_checkpoint, Logger, set_seed
from library.model import BirdResNet34


def apply_mixup(x, y, alpha, device):
    """
    Applies Mixup augmentation to a batch of data.

    Args:
        x (torch.Tensor): Input images.
        y (torch.Tensor): Input labels.
        alpha (float): Mixup alpha parameter.
        device (str): Device to move data to.

    Returns:
        tuple: (mixed_x, y_a, y_b, lam)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer.
        criterion (Loss): Loss function.
        device (str): Device.
        mixup_alpha (float): Alpha for mixup.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Apply Mixup
        if mixup_alpha > 0:
            images, targets_a, targets_b, lam = apply_mixup(
                images, labels, mixup_alpha, device
            )
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                outputs, targets_b
            )
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size if dataset_size > 0 else 0.0


def validate(model, loader, criterion, device):
    """
    Validates the model.

    Args:
        model (nn.Module): The model to validate.
        loader (DataLoader): Validation data loader.
        criterion (Loss): Loss function.
        device (str): Device.

    Returns:
        tuple: (avg_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        auc = compute_auc(all_targets, all_preds)
    else:
        auc = 0.0

    return avg_loss, auc


def run_training_cycle(
    model_name,
    train_loader,
    val_loader,
    mixup_alpha,
    swa_start_epoch,
    num_epochs=Config.NUM_EPOCHS,
    learning_rate=Config.LEARNING_RATE,
    swa_lr=Config.SWA_LR,
    device=Config.DEVICE,
):
    """
    Runs the full training cycle including SWA.

    Args:
        model_name (str): Name identifier for saving checkpoints.
        train_loader (DataLoader): Training loader.
        val_loader (DataLoader): Validation loader.
        mixup_alpha (float): Mixup intensity.
        swa_start_epoch (int): Epoch to start SWA.
        num_epochs (int): Total epochs.
        learning_rate (float): Initial LR.
        swa_lr (float): SWA LR.
        device (str): Device.

    Returns:
        nn.Module: The final SWA model (with updated BN).
    """
    set_seed(Config.SEED)
    logger = Logger(f"log_{model_name}.txt")
    logger.log(
        f"Starting training for {model_name} | Mixup: {mixup_alpha} | SWA Start: {swa_start_epoch}"
    )

    # 1. Initialize Model
    model = BirdResNet34(pretrained=True).to(device)

    # 2. Optimizer & Criterion
    # Using SGD with Momentum as standard for SWA
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=Config.WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    # 3. SWA Setup
    swa_model = AveragedModel(model).to(device)

    # Scheduler: Cosine Annealing for initial phase, SWALR for SWA phase
    # We decay from LR to SWA_LR by the time SWA starts
    scheduler = CosineAnnealingLR(optimizer, T_max=swa_start_epoch, eta_min=swa_lr)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

    best_auc = 0.0

    for epoch in range(num_epochs):
        # Determine if we are in SWA phase
        is_swa_phase = epoch >= swa_start_epoch

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, mixup_alpha
        )

        # Update SWA or Step Scheduler
        if is_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            lr = swa_scheduler.get_last_lr()[0]
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

        # Validate Base Model
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Logging
        logger.log(
            f"Epoch {epoch+1}/{num_epochs} [LR: {lr:.6f}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        # Save Best Base Model
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                filename=f"{model_name}_base_best.pth",
            )

    # Save Last Base Model
    save_checkpoint(
        {
            "epoch": num_epochs,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        filename=f"{model_name}_last.pth",
    )

    # 4. Finalize SWA
    logger.log("Updating SWA Batch Normalization statistics...")
    update_bn(train_loader, swa_model, device=device)

    # Validate SWA Model
    swa_val_loss, swa_val_auc = validate(swa_model, val_loader, criterion, device)
    logger.log(
        f"SWA Final Results | Val Loss: {swa_val_loss:.6f} | Val AUC: {swa_val_auc:.10f}"
    )

    # Save SWA Model
    save_checkpoint(
        {"epoch": num_epochs, "state_dict": swa_model.state_dict(), "auc": swa_val_auc},
        filename=f"{model_name}_swa.pth",
    )

    return swa_model
