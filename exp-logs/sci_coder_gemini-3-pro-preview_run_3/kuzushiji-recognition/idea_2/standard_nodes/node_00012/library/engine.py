import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from library.config import Config

# ==================================================================================
# 1. Custom Loss Functions
# ==================================================================================


class ModifiedFocalLoss(nn.Module):
    """
    Modified Focal Loss for CenterNet Heatmap.
    Penalizes deviations from the ground truth Gaussian heatmap.
    """

    def __init__(self, alpha=2, beta=4):
        super(ModifiedFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        Args:
            pred: (Batch, 1, H, W) - Predicted heatmap (sigmoid applied)
            gt:   (Batch, 1, H, W) - Ground truth heatmap with Gaussian peaks
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.beta)

        loss = 0

        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

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
    L1 Loss masked by the ground truth heatmap centers.
    Used for Size and Offset regression.
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, mask):
        """
        Args:
            pred:   (Batch, C, H, W)
            target: (Batch, C, H, W)
            mask:   (Batch, 1, H, W) - 1 at object centers, 0 otherwise
        """
        # Expand mask to match channel dimension of pred
        expand_mask = mask.repeat(1, pred.shape[1], 1, 1)

        loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction="sum")

        # Normalize by number of objects (sum of mask)
        # Add epsilon to prevent division by zero
        mask_sum = mask.sum() + 1e-4
        loss = loss / mask_sum

        return loss


# ==================================================================================
# 2. Detector Training Engine (Stage 1)
# ==================================================================================


def train_one_epoch_detector(
    model, loader, optimizer, criterion_hm, criterion_reg, device
):
    model.train()
    running_loss = 0.0
    running_hm_loss = 0.0
    running_wh_loss = 0.0
    running_off_loss = 0.0

    for batch in loader:
        if batch is None:
            continue

        imgs = batch["img"].to(device)
        hm_target = batch["heatmap"].to(device)
        wh_target = batch["size_map"].to(device)
        off_target = batch["offset_map"].to(device)

        # Forward
        outputs = model(imgs)
        pred_hm = outputs["heatmap"]
        pred_wh = outputs["size_map"]
        pred_off = outputs["offset_map"]

        # Mask for regression (where heatmap == 1)
        mask = hm_target.eq(1).float()

        # Losses
        loss_hm = criterion_hm(pred_hm, hm_target)
        loss_wh = criterion_reg(pred_wh, wh_target, mask)
        loss_off = criterion_reg(pred_off, off_target, mask)

        # Weights: Heatmap=1.0, Size=0.1, Offset=1.0
        loss = loss_hm + 0.1 * loss_wh + loss_off

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_hm_loss += loss_hm.item()
        running_wh_loss += loss_wh.item()
        running_off_loss += loss_off.item()

    num_batches = len(loader)
    return {
        "loss": running_loss / num_batches,
        "hm_loss": running_hm_loss / num_batches,
        "wh_loss": running_wh_loss / num_batches,
        "off_loss": running_off_loss / num_batches,
    }


def evaluate_detector(model, loader, criterion_hm, criterion_reg, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            imgs = batch["img"].to(device)
            hm_target = batch["heatmap"].to(device)
            wh_target = batch["size_map"].to(device)
            off_target = batch["offset_map"].to(device)

            outputs = model(imgs)
            pred_hm = outputs["heatmap"]
            pred_wh = outputs["size_map"]
            pred_off = outputs["offset_map"]

            mask = hm_target.eq(1).float()

            loss_hm = criterion_hm(pred_hm, hm_target)
            loss_wh = criterion_reg(pred_wh, wh_target, mask)
            loss_off = criterion_reg(pred_off, off_target, mask)

            loss = loss_hm + 0.1 * loss_wh + loss_off
            running_loss += loss.item()

    return running_loss / len(loader)


def train_detector(model, train_loader, val_loader, optimizer, device=Config.DEVICE):
    print(f"Starting Detector Training on {device}...")

    criterion_hm = ModifiedFocalLoss().to(device)
    criterion_reg = RegL1Loss().to(device)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.DETECTOR_EPOCHS):
        # Train
        train_metrics = train_one_epoch_detector(
            model, train_loader, optimizer, criterion_hm, criterion_reg, device
        )

        # Validate
        val_loss = evaluate_detector(
            model, val_loader, criterion_hm, criterion_reg, device
        )

        print(f"Epoch {epoch+1}/{Config.DETECTOR_EPOCHS}")
        print(f"Train Loss: {train_metrics['loss']}")
        print(
            f"Train Components - HM: {train_metrics['hm_loss']}, WH: {train_metrics['wh_loss']}, OFF: {train_metrics['off_loss']}"
        )
        print(f"Val Loss: {val_loss}")

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.DETECTOR_MODEL_PATH)
            print(f"New best model saved to {Config.DETECTOR_MODEL_PATH}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.DETECTOR_PATIENCE}"
            )

        if patience_counter >= Config.DETECTOR_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Detector training completed.")


# ==================================================================================
# 3. Classifier Training Engine (Stage 2)
# ==================================================================================


def train_one_epoch_classifier(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        if batch is None:
            continue

        imgs = batch["img"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    num_batches = len(loader)
    accuracy = correct / total if total > 0 else 0.0
    return running_loss / num_batches, accuracy


def evaluate_classifier(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            imgs = batch["img"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    num_batches = len(loader)
    accuracy = correct / total if total > 0 else 0.0
    return running_loss / num_batches, accuracy


def train_classifier(model, train_loader, val_loader, optimizer, device=Config.DEVICE):
    print(f"Starting Classifier Training on {device}...")

    # Use standard CrossEntropyLoss
    # Note: If class imbalance is severe, one might use weighted CE here,
    # but for this implementation we stick to standard CE as per plan.
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.CLASSIFIER_EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch_classifier(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_acc = evaluate_classifier(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{Config.CLASSIFIER_EPOCHS}")
        print(f"Train Loss: {train_loss}, Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss}, Val Acc: {val_acc}")

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.CLASSIFIER_MODEL_PATH)
            print(f"New best model saved to {Config.CLASSIFIER_MODEL_PATH}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.CLASSIFIER_PATIENCE}"
            )

        if patience_counter >= Config.CLASSIFIER_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Classifier training completed.")
