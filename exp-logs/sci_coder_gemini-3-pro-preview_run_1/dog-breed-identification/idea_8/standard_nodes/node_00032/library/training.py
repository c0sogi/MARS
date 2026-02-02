import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import seed_everything, calculate_metric, SWAHandler, Logger
from library.dataset import get_dataloaders
from library.models import create_model

# -----------------------------------------------------------------------------
# Mixup / CutMix Utilities
# -----------------------------------------------------------------------------


def rand_bbox(size, lam):
    """Generates a random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0, device="cuda"):
    """Returns mixed inputs (CutMix), pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# -----------------------------------------------------------------------------
# Core Training Functions
# -----------------------------------------------------------------------------


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    use_mixup=False,
    mixup_prob=0.0,
    debug=False,
):
    model.train()
    running_loss = 0.0
    scaler = GradScaler()

    for i, (images, targets, _) in enumerate(loader):
        if debug and i >= 10:
            break  # Limit batches in debug mode

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Mixup / CutMix Logic
        apply_mix = False
        if use_mixup and np.random.rand() < mixup_prob:
            apply_mix = True
            if np.random.rand() < 0.5:
                # Mixup
                images, y_a, y_b, lam = mixup_data(
                    images, targets, Config.MIXUP_ALPHA, device
                )
            else:
                # CutMix
                images, y_a, y_b, lam = cutmix_data(
                    images, targets, Config.CUTMIX_ALPHA, device
                )

        with autocast():
            outputs = model(images)
            if apply_mix:
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device, debug=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (images, targets, _) in enumerate(loader):
            if debug and i >= 10:
                break

            images = images.to(device)
            targets = targets.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    total_loss = (
        running_loss / len(loader.dataset)
        if not debug
        else running_loss / (10 * Config.BATCH_SIZE)
    )

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metric = calculate_metric(all_targets, all_preds)

    return total_loss, metric


# -----------------------------------------------------------------------------
# Regime A: ConvNeXt (Precision Track)
# -----------------------------------------------------------------------------


def run_regime_a(fold_idx, debug=False):
    seed_everything(Config.SEED + fold_idx)

    log_path = os.path.join(Config.WORK_DIR, f"train_regime_a_fold_{fold_idx}.log")
    logger = Logger(log_path)
    logger.log(f"Starting Regime A (ConvNeXt) for Fold {fold_idx}")

    device = torch.device(Config.DEVICE)
    train_loader, val_loader = get_dataloaders(fold_idx)

    model = create_model(
        Config.MODEL_A_NAME, num_classes=Config.NUM_CLASSES, pretrained=True
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # --- Phase 1: Head Adaptation (Frozen Backbone) ---
    logger.log("Phase 1: Head Adaptation")

    # Freeze backbone
    # In timm models, classifier is usually 'head' or 'fc'
    # We use get_classifier() to identify head params
    head_params = list(map(id, model.get_classifier().parameters()))
    for name, param in model.named_parameters():
        if id(param) not in head_params:
            param.requires_grad = False

    optimizer = optim.AdamW(
        model.get_classifier().parameters(), lr=Config.REGIME_A_PHASE1_LR
    )

    epochs_p1 = 1 if debug else Config.REGIME_A_PHASE1_EPOCHS
    for epoch in range(epochs_p1):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, debug=debug
        )
        val_loss, val_metric = validate(
            model, val_loader, criterion, device, debug=debug
        )
        logger.log(
            f"[P1] Epoch {epoch+1}: Train Loss={loss:.4f}, Val Loss={val_loss:.6f}, Val Metric={val_metric:.6f}"
        )

    # --- Phase 2: Fine-Tuning (Discriminative LRs) ---
    logger.log("Phase 2: Fine-Tuning with Discriminative LRs")

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    # Parameter groups
    backbone_params = [
        p for n, p in model.named_parameters() if id(p) not in head_params
    ]
    head_params_list = list(model.get_classifier().parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.REGIME_A_BACKBONE_LR},
            {"params": head_params_list, "lr": Config.REGIME_A_HEAD_LR},
        ]
    )

    epochs_p2 = 1 if debug else Config.REGIME_A_PHASE2_EPOCHS
    for epoch in range(epochs_p2):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, debug=debug
        )
        val_loss, val_metric = validate(
            model, val_loader, criterion, device, debug=debug
        )
        logger.log(
            f"[P2] Epoch {epoch+1}: Train Loss={loss:.4f}, Val Loss={val_loss:.6f}, Val Metric={val_metric:.6f}"
        )

    # --- Phase 3: SWA ---
    logger.log("Phase 3: Stochastic Weight Averaging")

    swa_handler = SWAHandler(model, device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.REGIME_A_SWA_LR)

    epochs_p3 = 1 if debug else Config.REGIME_A_SWA_EPOCHS
    for epoch in range(epochs_p3):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, debug=debug
        )
        swa_handler.update(model)

        # Evaluate current model (optional, just for tracking)
        val_loss, val_metric = validate(
            model, val_loader, criterion, device, debug=debug
        )
        logger.log(
            f"[P3] Epoch {epoch+1}: Train Loss={loss:.4f}, Val Loss={val_loss:.6f}, Val Metric={val_metric:.6f}"
        )

    # Update BN statistics for SWA model
    logger.log("Updating SWA Batch Norm statistics...")
    swa_handler.update_bn(train_loader)

    # Final Evaluation of SWA model
    final_model = swa_handler.get_model()
    val_loss, val_metric = validate(
        final_model, val_loader, criterion, device, debug=debug
    )
    logger.log(f"Final SWA Model: Val Loss={val_loss:.6f}, Val Metric={val_metric:.6f}")

    # Save SWA model
    save_path = os.path.join(Config.WORK_DIR, f"convnext_base_fold_{fold_idx}.pth")
    torch.save(final_model.state_dict(), save_path)
    logger.log(f"Saved model to {save_path}")
