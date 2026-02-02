import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from library.config import Config

# ==========================================
# 0. Setup & Reproducibility
# ==========================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(Config.SEED)

# ==========================================
# 1. Loss Functions
# ==========================================


class ModifiedFocalLoss(nn.Module):
    """
    Modified Focal Loss for Heatmap regression (from CenterNet/CornerNet).
    """

    def __init__(self, alpha=2, beta=4):
        super(ModifiedFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        pred: (batch, c, h, w)
        gt:   (batch, c, h, w)
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.beta)

        loss = 0

        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, 1e-12, 1 - 1e-12)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = (
            torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss


class RegL1Loss(nn.Module):
    """
    L1 Loss masked by object presence.
    Used for Size (wh) and Offset (reg) heads.
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, mask):
        """
        pred:   (batch, 2, h, w)
        target: (batch, 2, h, w)
        mask:   (batch, 1, h, w)
        """
        # Expand mask to match channel dim of pred
        mask = mask.expand_as(pred)

        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")

        # Normalize by number of objects (sum of mask) + epsilon
        loss = loss / (mask.sum() + 1e-4)
        return loss


# ==========================================
# 2. Utilities
# ==========================================


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ==========================================
# 3. Detector Engine
# ==========================================


def train_detector_one_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()

    losses = AverageMeter()
    hm_losses = AverageMeter()
    wh_losses = AverageMeter()
    off_losses = AverageMeter()

    hm_criterion = ModifiedFocalLoss()
    l1_criterion = RegL1Loss()

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        hm_target = batch["hm"].to(device)
        wh_target = batch["wh"].to(device)
        reg_target = batch["reg"].to(device)
        reg_mask = batch["reg_mask"].to(device)

        optimizer.zero_grad()

        with autocast():
            hm_pred, wh_pred, reg_pred = model(images)

            loss_hm = hm_criterion(hm_pred, hm_target)
            loss_wh = l1_criterion(wh_pred, wh_target, reg_mask)
            loss_off = l1_criterion(reg_pred, reg_target, reg_mask)

            loss = (
                Config.LOSS_HEATMAP_WEIGHT * loss_hm
                + Config.LOSS_SIZE_WEIGHT * loss_wh
                + Config.LOSS_OFFSET_WEIGHT * loss_off
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), images.size(0))
        hm_losses.update(loss_hm.item(), images.size(0))
        wh_losses.update(loss_wh.item(), images.size(0))
        off_losses.update(loss_off.item(), images.size(0))

    print(
        f"Detector Train Epoch [{epoch}] Loss: {losses.avg:.6f} "
        f"(HM: {hm_losses.avg:.6f}, WH: {wh_losses.avg:.6f}, Off: {off_losses.avg:.6f})"
    )

    return losses.avg


def validate_detector(model, loader, device):
    model.eval()

    losses = AverageMeter()
    hm_losses = AverageMeter()
    wh_losses = AverageMeter()
    off_losses = AverageMeter()

    hm_criterion = ModifiedFocalLoss()
    l1_criterion = RegL1Loss()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            hm_target = batch["hm"].to(device)
            wh_target = batch["wh"].to(device)
            reg_target = batch["reg"].to(device)
            reg_mask = batch["reg_mask"].to(device)

            with autocast():
                hm_pred, wh_pred, reg_pred = model(images)

                loss_hm = hm_criterion(hm_pred, hm_target)
                loss_wh = l1_criterion(wh_pred, wh_target, reg_mask)
                loss_off = l1_criterion(reg_pred, reg_target, reg_mask)

                loss = (
                    Config.LOSS_HEATMAP_WEIGHT * loss_hm
                    + Config.LOSS_SIZE_WEIGHT * loss_wh
                    + Config.LOSS_OFFSET_WEIGHT * loss_off
                )

            losses.update(loss.item(), images.size(0))
            hm_losses.update(loss_hm.item(), images.size(0))
            wh_losses.update(loss_wh.item(), images.size(0))
            off_losses.update(loss_off.item(), images.size(0))

    print(
        f"Detector Val Loss: {losses.avg:.6f} "
        f"(HM: {hm_losses.avg:.6f}, WH: {wh_losses.avg:.6f}, Off: {off_losses.avg:.6f})"
    )

    return losses.avg


def fit_detector(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=Config.DETECTOR_EPOCHS,
    patience=5,
):
    scaler = GradScaler()
    best_loss = float("inf")
    patience_counter = 0
    save_path = os.path.join(Config.WORKING_DIR, "detector_best.pth")

    print(f"Starting Detector Training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_loss = train_detector_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch
        )
        val_loss = validate_detector(model, val_loader, device)

        # Checkpointing & Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(
                f"New best detector model saved to {save_path} (Loss: {best_loss:.6f})"
            )
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("Loaded best detector model.")

    return model


# ==========================================
# 4. Classifier Engine
# ==========================================


def train_classifier_one_epoch(
    model, loader, criterion, optimizer, scaler, device, epoch
):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()

        losses.update(loss.item(), images.size(0))
        accuracies.update(acc.item(), images.size(0))

    print(
        f"Classifier Train Epoch [{epoch}] Loss: {losses.avg:.6f} Acc: {accuracies.avg:.6f}"
    )
    return losses.avg, accuracies.avg


def validate_classifier(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            acc = (preds == labels).float().mean()

            losses.update(loss.item(), images.size(0))
            accuracies.update(acc.item(), images.size(0))

    print(f"Classifier Val Loss: {losses.avg:.6f} Acc: {accuracies.avg:.6f}")
    return losses.avg, accuracies.avg


def fit_classifier(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=Config.CLASSIFIER_EPOCHS,
    patience=3,
):
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    patience_counter = 0
    save_path = os.path.join(Config.WORKING_DIR, "classifier_best.pth")

    print(f"Starting Classifier Training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_classifier_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_loss, val_acc = validate_classifier(model, val_loader, criterion, device)

        # Checkpointing & Early Stopping based on Accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(
                f"New best classifier model saved to {save_path} (Acc: {best_acc:.6f})"
            )
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("Loaded best classifier model.")

    return model
