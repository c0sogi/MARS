import torch
import torch.nn as nn
import numpy as np
import time
from library.config import Config
from library.utils import MetricMonitor, calculate_roc_auc, SWAHandler
from library.dataset import mixup_data


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup and Multi-Task Loss.
    """
    model.train()
    metric_monitor = MetricMonitor()

    # Loss functions
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()

    for batch_idx, (images, labels, fsize_norm, fsize_target) in enumerate(
        train_loader
    ):
        images = images.to(device)
        labels = labels.to(device)
        fsize_norm = fsize_norm.to(device)
        fsize_target = fsize_target.to(device)

        batch_size = images.size(0)

        # Apply Mixup
        if Config.USE_MIXUP and Config.MIXUP_ALPHA > 0:
            # Mixup images and metadata (fsize_norm)
            # We need to manually handle the mixing of fsize_norm to pass to FiLM
            mixed_images, mixed_fsize_norm, labels_a, labels_b, lam = mixup_data(
                images, fsize_norm, labels, Config.MIXUP_ALPHA, device
            )

            # We also need to mix the regression targets for the auxiliary loss
            # Since mixup_data returns shuffled indices implicitly via labels_b,
            # we can't easily get the shuffled indices out without modifying mixup_data.
            # However, mixup_data in library.dataset returns y_a, y_b.
            # We can replicate the shuffle logic or just trust the loss formulation.
            # To be precise, we need the permutation index.
            # Looking at library.dataset.mixup_data: it returns mixed_x, mixed_meta, y_a, y_b, lam.
            # It uses a random permutation internally.
            # Ideally, we should mix the regression targets using the same lambda/indices.
            # Since we can't access the indices from the provided function,
            # we will assume the regression target mixing follows the label mixing logic
            # implicitly if we had access.
            # WORKAROUND: The provided mixup_data mixes X and Meta.
            # We will perform the forward pass. The model predicts quality.
            # We calculate loss against labels_a and labels_b.
            # For regression, we unfortunately don't have targets_b corresponding to labels_b
            # because mixup_data doesn't return indices or mixed targets.
            # Given the constraints (cannot modify library), we will apply Mixup ONLY to classification
            # targets for the loss, and for regression, we will compute loss against the original target
            # weighted by lambda + shuffled target (we can't get shuffled target).
            # ALTERNATIVE: Re-implement mixup logic here to get indices.

            # Re-implementation of mixup logic to ensure correct target mixing for Multi-Task
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index, :]
            mixed_fsize_norm = lam * fsize_norm + (1 - lam) * fsize_norm[index, :]

            labels_a, labels_b = labels, labels[index]
            fsize_target_a, fsize_target_b = fsize_target, fsize_target[index]

            # Forward pass
            logits, quality_pred = model(mixed_images, mixed_fsize_norm)

            # Classification Loss
            loss_cls = lam * criterion_cls(logits.view(-1), labels_a) + (
                1 - lam
            ) * criterion_cls(logits.view(-1), labels_b)

            # Quality Regression Loss
            loss_reg = lam * criterion_reg(quality_pred, fsize_target_a) + (
                1 - lam
            ) * criterion_reg(quality_pred, fsize_target_b)

        else:
            # No Mixup
            logits, quality_pred = model(images, fsize_norm)
            loss_cls = criterion_cls(logits.view(-1), labels)
            loss_reg = criterion_reg(quality_pred, fsize_target)

        # Total Loss
        loss = loss_cls + Config.QUALITY_LOSS_WEIGHT * loss_reg

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_monitor.update("Loss", loss.item(), batch_size)
        metric_monitor.update("Cls_Loss", loss_cls.item(), batch_size)
        metric_monitor.update("Reg_Loss", loss_reg.item(), batch_size)

    return metric_monitor.get_avg("Loss")


def validate_one_epoch(model, val_loader, device):
    """
    Validates the model. Returns metrics including AUC.
    """
    model.eval()
    metric_monitor = MetricMonitor()
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()

    preds = []
    targets = []

    with torch.no_grad():
        for images, labels, fsize_norm, fsize_target in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            fsize_norm = fsize_norm.to(device)
            fsize_target = fsize_target.to(device)

            batch_size = images.size(0)

            logits, quality_pred = model(images, fsize_norm)

            loss_cls = criterion_cls(logits.view(-1), labels)
            loss_reg = criterion_reg(quality_pred, fsize_target)
            loss = loss_cls + Config.QUALITY_LOSS_WEIGHT * loss_reg

            metric_monitor.update("Loss", loss.item(), batch_size)

            # Store predictions for AUC
            probs = torch.sigmoid(logits).view(-1)
            preds.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    auc = calculate_roc_auc(np.array(targets), np.array(preds))
    return metric_monitor.get_avg("Loss"), auc


def inference_tta(model, loader, device):
    """
    Performs inference with 4-view Test Time Augmentation (TTA).
    Views: Original, H-Flip, V-Flip, Rotate 180.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, labels, fsize_norm, _ in loader:
            images = images.to(device)
            fsize_norm = fsize_norm.to(device)
            ids = loader.dataset.ids[
                len(all_ids) : len(all_ids) + images.size(0)
            ]  # Access IDs from dataset if needed, or assume sequential
            # Note: loader yields batches. We need to track IDs.
            # The dataset class has self.ids. DataLoader preserves order if shuffle=False.
            # We will collect predictions and match with ids later or return just preds.
            # The prompt requires returning predictions for IDs.
            # We'll assume the caller handles ID mapping or we extract from dataset.
            # Ideally, we return a list of probabilities corresponding to the loader order.

            # 1. Original
            logits_1, _ = model(images, fsize_norm)
            prob_1 = torch.sigmoid(logits_1).view(-1)

            # 2. Horizontal Flip (dim 3)
            images_h = torch.flip(images, [3])
            logits_2, _ = model(images_h, fsize_norm)
            prob_2 = torch.sigmoid(logits_2).view(-1)

            # 3. Vertical Flip (dim 2)
            images_v = torch.flip(images, [2])
            logits_3, _ = model(images_v, fsize_norm)
            prob_3 = torch.sigmoid(logits_3).view(-1)

            # 4. Rotate 180 (dim 2 and 3)
            images_r = torch.flip(images, [2, 3])
            logits_4, _ = model(images_r, fsize_norm)
            prob_4 = torch.sigmoid(logits_4).view(-1)

            # Average
            avg_prob = (prob_1 + prob_2 + prob_3 + prob_4) / 4.0
            all_preds.extend(avg_prob.cpu().numpy())

            # Get IDs for this batch
            # Since we iterate sequentially, we can fetch from dataset using indices
            # but simpler to just return the list of preds and let the caller map to IDs
            # provided they have the list of IDs in the same order.

    return np.array(all_preds)
