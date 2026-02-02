import torch
import torch.nn as nn
import numpy as np
import itertools
from library.config import Config
from library.utils import calculate_iou_batch


def train_teacher_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the teacher model for one epoch using Multi-Task Learning and Bernoulli Injection Masking.

    Args:
        model: The PyTorch model.
        loader: DataLoader for labeled training data.
        optimizer: Optimizer.
        criterion: Loss function (CombinedMTLLoss).
        device: Device to train on.
        epoch: Current epoch number.

    Returns:
        dict: Average loss metrics for the epoch.
    """
    model.train()
    metrics_sum = {}
    num_batches = 0

    # Bernoulli Drop Probability
    drop_prob = Config.BERNOULLI_DROP_PROB

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        depths = batch["depth"].to(device)

        batch_size = images.size(0)

        # --- Bernoulli Injection Masking ---
        # Generate random mask: 1 means drop (use 0), 0 means keep (use true depth)
        drop_mask = torch.bernoulli(
            torch.full((batch_size, 1), drop_prob, device=device)
        )

        # If drop_mask is 1, input_depth becomes 0.0 (Mean). If 0, it remains original depth.
        input_depths = depths * (1.0 - drop_mask)

        optimizer.zero_grad()

        # Forward Pass
        outputs = model(images, input_depths)

        # Loss Calculation
        # Targets: 'mask' is ground truth. 'depth' is ALWAYS ground truth (Auxiliary Consistency)
        targets = {"mask": masks, "depth": depths}

        loss, batch_metrics = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # Accumulate metrics
        num_batches += 1
        for k, v in batch_metrics.items():
            val = v.item()
            if k not in metrics_sum:
                metrics_sum[k] = 0.0
            metrics_sum[k] += val

    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    print(f"Epoch {epoch} [Teacher] - Loss: {avg_metrics['loss_total']}")
    return avg_metrics


def train_student_epoch(
    model, labeled_loader, unlabeled_loader, optimizer, criterion, device, epoch
):
    """
    Trains the student model for one epoch using Noisy Student Distillation.

    Args:
        model: The PyTorch model.
        labeled_loader: DataLoader for labeled data (Image + True Depth).
        unlabeled_loader: DataLoader for unlabeled data (Augmented Image + Zero Depth + Soft Labels).
        optimizer: Optimizer.
        criterion: Loss function (CombinedMTLLoss).
        device: Device to train on.
        epoch: Current epoch number.

    Returns:
        dict: Average loss metrics.
    """
    model.train()
    metrics_sum = {}
    num_batches = 0

    # Cycle through unlabeled loader to match labeled loader length
    unlabeled_iter = itertools.cycle(unlabeled_loader)

    for labeled_batch in labeled_loader:
        # Get labeled data
        l_images = labeled_batch["image"].to(device)
        l_masks = labeled_batch["mask"].to(device)
        l_depths = labeled_batch["depth"].to(device)

        # Get unlabeled data
        try:
            u_batch = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = itertools.cycle(unlabeled_loader)
            u_batch = next(unlabeled_iter)

        u_images = u_batch["image"].to(device)
        u_soft_masks = u_batch["mask"].to(device)  # Soft pseudo-labels
        u_depths = u_batch["depth"].to(device)  # Zeros (Generalist Mode)

        optimizer.zero_grad()

        # --- Labeled Step ---
        # Standard MTL training with True Depth
        l_outputs = model(l_images, l_depths)
        l_targets = {"mask": l_masks, "depth": l_depths}
        l_loss, l_metrics = criterion(l_outputs, l_targets)

        # --- Unlabeled Step ---
        # Distillation: BCE against soft targets
        # Input depth is forced to zero (Generalist Mode)
        u_outputs = model(u_images, u_depths)
        u_pred_mask = u_outputs["mask"]

        # Use BCE part of criterion for soft labels
        u_loss = criterion.bce(u_pred_mask, u_soft_masks)

        # Combine Losses
        total_loss = l_loss + u_loss

        total_loss.backward()
        optimizer.step()

        # Accumulate metrics
        num_batches += 1

        current_metrics = {
            "loss_labeled": l_loss.item(),
            "loss_unlabeled": u_loss.item(),
            "loss_total": total_loss.item(),
        }

        for k, v in current_metrics.items():
            if k not in metrics_sum:
                metrics_sum[k] = 0.0
            metrics_sum[k] += v

    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
    print(f"Epoch {epoch} [Student] - Total Loss: {avg_metrics['loss_total']}")
    return avg_metrics


def validate(model, loader, criterion, device, force_zero_depth=True):
    """
    Validates the model.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device.
        force_zero_depth (bool): If True, replaces depth with 0.0 to simulate test conditions.

    Returns:
        dict: Validation metrics (loss, mAP).
    """
    model.eval()
    metrics_sum = {}
    num_batches = 0

    # For mAP calculation
    iou_thresholds = Config.IOU_THRESHOLDS
    map_score_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            depths = batch["depth"].to(device)

            # Determine input depth (Generalist vs Specialist check)
            if force_zero_depth:
                input_depths = torch.zeros_like(depths)
            else:
                input_depths = depths

            outputs = model(images, input_depths)

            # Loss Calculation (Aux head always targets true depth)
            targets = {"mask": masks, "depth": depths}
            loss, batch_metrics = criterion(outputs, targets)

            # Accumulate Loss Metrics
            num_batches += 1
            for k, v in batch_metrics.items():
                val = v.item()
                if k not in metrics_sum:
                    metrics_sum[k] = 0.0
                metrics_sum[k] += val

            # --- Calculate mAP ---
            preds = torch.sigmoid(outputs["mask"])

            # Calculate IoU at each threshold for the batch
            batch_ious = []
            for t in iou_thresholds:
                iou = calculate_iou_batch(preds, masks, threshold=t)  # (B,)
                batch_ious.append(iou)

            # Stack to (n_thresh, B)
            batch_ious = np.stack(batch_ious, axis=0)

            # Compare against thresholds
            t_col = iou_thresholds[:, np.newaxis]  # (n_thresh, 1)
            is_hit = (batch_ious > t_col).astype(np.float32)  # (n_thresh, B)

            # Average Precision per image is mean over thresholds
            image_aps = is_hit.mean(axis=0)  # (B,)

            map_score_sum += image_aps.sum()

    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    # Calculate global mAP
    total_samples = len(loader.dataset)
    mAP = map_score_sum / total_samples

    avg_metrics["mAP"] = mAP

    print(f"Validation Results - Loss: {avg_metrics['loss_total']} | mAP: {mAP}")
    return avg_metrics


def predict(model, loader, device):
    """
    Generates predictions for the test set using TTA (Horizontal Flip).

    Args:
        model: The PyTorch model.
        loader: DataLoader for test data.
        device: Device.

    Returns:
        dict: Dictionary mapping image ID to predicted probability map (np.ndarray).
    """
    model.eval()
    predictions = {}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            depths = batch["depth"].to(device)  # Should be zeros for test
            ids = batch["id"]

            # TTA: Original
            out_orig = model(images, depths)["mask"]
            prob_orig = torch.sigmoid(out_orig)

            # TTA: Flip
            images_flip = torch.flip(images, [3])  # Flip width (dim 3)
            out_flip = model(images_flip, depths)["mask"]
            prob_flip = torch.sigmoid(out_flip)
            prob_flip_back = torch.flip(prob_flip, [3])

            # Average
            prob_avg = (prob_orig + prob_flip_back) / 2.0

            # To Numpy
            prob_avg = prob_avg.cpu().numpy()  # (B, 1, H, W)

            for i, img_id in enumerate(ids):
                # Squeeze channel
                pred_map = prob_avg[i, 0, :, :]
                predictions[img_id] = pred_map

    return predictions
