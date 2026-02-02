import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import AverageMeter, calculate_roc_auc
from library.data import mixup_data, mixup_criterion


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch using Mixup and Multi-Head Loss.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        # Reshape targets to (B, 1) for BCEWithLogitsLoss
        targets = targets.to(device).view(-1, 1)

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, targets, Config.MIXUP_ALPHA, device
        )

        # Forward pass (returns logits from both heads)
        tex_logits, sem_logits = model(images)

        # Compute Loss for both heads
        loss_tex = mixup_criterion(criterion, tex_logits, targets_a, targets_b, lam)
        loss_sem = mixup_criterion(criterion, sem_logits, targets_a, targets_b, lam)

        # Total loss is the unweighted sum
        loss = loss_tex + loss_sem

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC.
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).view(-1, 1)

            # Forward pass
            tex_logits, sem_logits = model(images)

            # Compute Loss
            loss_tex = criterion(tex_logits, targets)
            loss_sem = criterion(sem_logits, targets)
            loss = loss_tex + loss_sem

            losses.update(loss.item(), images.size(0))

            # Compute probabilities (Sigmoid) and average heads
            probs_tex = torch.sigmoid(tex_logits)
            probs_sem = torch.sigmoid(sem_logits)
            avg_probs = (probs_tex + probs_sem) / 2.0

            all_preds.extend(avg_probs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    # Calculate ROC AUC
    auc = calculate_roc_auc(np.array(all_targets), np.array(all_preds))
    return losses.avg, auc


def train_fold_swa(model, train_loader, val_loader, fold_idx, device):
    """
    Orchestrates the SWA training pipeline for a single fold.
    Phase 1: Convergence (AdamW + CosineAnnealing)
    Phase 2: SWA Exploration (SWALR + Averaging)
    """
    print(f"\n[Fold {fold_idx}] Starting Training...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR_MAX, weight_decay=Config.WEIGHT_DECAY
    )

    # --- Phase 1: Convergence ---
    print(
        f"[Fold {fold_idx}] Phase 1: Convergence ({Config.EPOCHS_CONVERGENCE} Epochs)"
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS_CONVERGENCE, eta_min=Config.LR_MIN
    )

    best_auc = 0.0

    for epoch in range(Config.EPOCHS_CONVERGENCE):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"  Epoch {epoch+1}/{Config.EPOCHS_CONVERGENCE} | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            # We could save the best convergence model here if needed
            # torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, f"best_fold{fold_idx}.pth"))

    # --- Phase 2: SWA Exploration ---
    print(f"[Fold {fold_idx}] Phase 2: SWA Exploration ({Config.EPOCHS_SWA} Epochs)")

    # Initialize SWA
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

    for epoch in range(Config.EPOCHS_SWA):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Update SWA Model
        swa_model.update_parameters(model)
        swa_scheduler.step()

        # Optional: Monitor current model performance
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        print(
            f"  SWA Epoch {epoch+1}/{Config.EPOCHS_SWA} | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.6f}"
        )

    # --- Finalization ---
    print(f"[Fold {fold_idx}] Updating BN statistics for SWA model...")
    # Update BatchNorm statistics using the training data
    update_bn(train_loader, swa_model, device=device)

    # Evaluate Final SWA Model
    swa_val_loss, swa_val_auc = evaluate(swa_model, val_loader, criterion, device)
    print(
        f"[Fold {fold_idx}] Final SWA Model | Val Loss: {swa_val_loss:.5f} | Val AUC: {swa_val_auc:.6f}"
    )

    # Save the underlying module (with averaged weights)
    # swa_model.module contains the actual model architecture with averaged parameters
    save_path = os.path.join(Config.CHECKPOINT_DIR, f"swa_fold{fold_idx}.pth")
    torch.save(swa_model.module.state_dict(), save_path)
    print(f"Saved SWA model to {save_path}")

    return swa_model.module


def predict_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (TTA).
    Augmentations: Original, HFlip, VFlip, HVFlip (180 Rotation).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # Create TTA batch: Stack 4 views along batch dimension
            # Shape: (B, C, H, W) -> (4*B, C, H, W)
            img_orig = images
            img_h = torch.flip(images, dims=[-1])
            img_v = torch.flip(images, dims=[-2])
            img_hv = torch.flip(images, dims=[-1, -2])

            # Stack inputs
            tta_batch = torch.cat([img_orig, img_h, img_v, img_hv], dim=0)

            # Forward pass on stacked batch
            tex_logits, sem_logits = model(tta_batch)

            # Compute probabilities
            probs = (torch.sigmoid(tex_logits) + torch.sigmoid(sem_logits)) / 2.0

            # Reshape back to (4, B, 1) to average across augmentations
            # tta_batch was constructed as [orig, h, v, hv], so we split by batch size
            batch_size = images.size(0)
            probs = probs.view(4, batch_size, -1)

            # Average across the 4 views
            avg_probs = probs.mean(dim=0)  # (B, 1)

            all_preds.extend(avg_probs.cpu().numpy().flatten())

    return np.array(all_preds)
