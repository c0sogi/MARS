import os
import numpy as np
import torch
import pandas as pd
from itertools import cycle

from library.config import Config
from library.utils import calc_map_score, unpad_image, rle_encode, optimize_threshold
from library.losses import calc_combined_loss


def train_epoch_teacher(model, loader, optimizer, device):
    """
    Trains the Specialist Teacher for one epoch using explicit depth injection.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Teacher expects: image, mask, depth
        images, masks, depths = batch
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()

        # Forward pass with depth injection
        # Teacher model signature: forward(x, z)
        logits = model(images, depths)

        # Loss calculation (Hard targets: Lovasz + BCE)
        loss = calc_combined_loss(logits, masks, soft_targets=False)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_epoch_student(model, labeled_loader, unlabeled_loader, optimizer, device):
    """
    Trains the Generalist Student for one epoch using semi-supervised learning.
    Combines labeled data (Multi-task) and unlabeled data (Distillation).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Cycle unlabeled loader to match labeled loader length
    unlabeled_iter = None
    if unlabeled_loader is not None:
        unlabeled_iter = cycle(unlabeled_loader)

    for labeled_batch in labeled_loader:
        l_images, l_masks, l_depths = labeled_batch
        l_images = l_images.to(device)
        l_masks = l_masks.to(device)
        l_depths = l_depths.to(device)

        optimizer.zero_grad()

        # 1. Labeled Step (Multi-task: Mask + Aux Depth)
        # Student returns (logits, depth_pred) in training mode
        l_logits, l_pred_depth = model(l_images)

        loss_labeled = calc_combined_loss(
            l_logits,
            l_masks,
            pred_depth=l_pred_depth,
            target_depth=l_depths,
            soft_targets=False,
        )

        # 2. Unlabeled Step (Distillation: Soft Mask)
        loss_unlabeled = 0.0
        if unlabeled_iter is not None:
            # Unlabeled loader returns (image, soft_mask)
            u_images, u_masks = next(unlabeled_iter)
            u_images = u_images.to(device)
            u_masks = u_masks.to(device)

            # Student returns (logits, depth_pred), we ignore depth for unlabeled
            u_logits, _ = model(u_images)

            # Loss vs Soft Targets (BCE only)
            loss_unlabeled = calc_combined_loss(u_logits, u_masks, soft_targets=True)

        # Combine losses
        loss = loss_labeled + loss_unlabeled

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate_epoch(model, loader, device, is_teacher=False):
    """
    Validates the model and calculates mAP on the validation set.
    Handles unpadding to ensure metrics are calculated on original image dimensions.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            # Loader returns (image, mask, depth)
            images, masks, depths = batch
            images = images.to(device)

            if is_teacher:
                depths = depths.to(device)
                logits = model(images, depths)
            else:
                # Student in eval mode returns only logits
                logits = model(images)

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(logits)

            # Move to CPU for metric calculation
            preds_np = preds.cpu().numpy()
            masks_np = masks.numpy()

            # Unpad to original size for accurate metrics
            for p, t in zip(preds_np, masks_np):
                # Remove channel dim if present: (1, H, W) -> (H, W)
                if p.ndim == 3:
                    p = p[0]
                if t.ndim == 3:
                    t = t[0]

                p_orig = unpad_image(p, Config.ORIG_SIZE)
                t_orig = unpad_image(t, Config.ORIG_SIZE)

                all_preds.append(p_orig)
                all_targets.append(t_orig)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate mAP
    score = calc_map_score(all_preds, all_targets)
    return score


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    save_path,
    is_teacher=True,
    unlabeled_loader=None,
    patience=10,
):
    """
    Main training loop with Early Stopping and Metric Tracking.
    """
    best_score = -1.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        if is_teacher:
            train_loss = train_epoch_teacher(model, train_loader, optimizer, device)
        else:
            train_loss = train_epoch_student(
                model, train_loader, unlabeled_loader, optimizer, device
            )

        val_score = validate_epoch(model, val_loader, device, is_teacher=is_teacher)

        if scheduler:
            scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.8f} | Val mAP: {val_score:.16f}"
        )

        # Checkpoint logic
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model


def predict_teacher_marginalized(model, loader, device, scan_depths):
    """
    Generates marginalized soft predictions using the Teacher model.
    Averages predictions across a range of depth values to handle depth uncertainty.

    Args:
        scan_depths (list): List of normalized depth values to scan.

    Returns:
        dict: Mapping of image_id to soft probability mask (numpy array).
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for batch in loader:
            # Test loader returns (image, id)
            images, ids = batch
            images = images.to(device)

            # Accumulate probabilities
            # Output shape: (B, 1, H, W)
            avg_preds = torch.zeros(
                (images.size(0), 1, images.size(2), images.size(3)), device=device
            )

            for z_val in scan_depths:
                # Create constant depth tensor for the batch
                z_tensor = torch.full((images.size(0), 1), z_val, device=device)

                logits = model(images, z_tensor)
                probs = torch.sigmoid(logits)
                avg_preds += probs

            # Average across scans
            avg_preds /= len(scan_depths)

            # Store results (keep padded size for student training alignment)
            preds_np = avg_preds.cpu().numpy()

            for i, pred in zip(ids, preds_np):
                results[i] = pred[0]  # Store as (H, W)

    return results


def generate_submission(model, test_loader, val_loader, device, output_path):
    """
    Generates the final submission file using the Student model.
    Optimizes the binarization threshold on the validation set first.
    """
    model.eval()

    # 1. Optimize Threshold on Validation Set
    print("Optimizing threshold on validation set...")
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            imgs, masks, _ = batch
            imgs = imgs.to(device)

            logits = model(imgs)
            preds = torch.sigmoid(logits)

            # Unpad
            preds_np = preds.cpu().numpy()
            masks_np = masks.numpy()

            for p, t in zip(preds_np, masks_np):
                p_orig = unpad_image(p[0], Config.ORIG_SIZE)
                t_orig = unpad_image(t[0], Config.ORIG_SIZE)
                val_preds.append(p_orig)
                val_targets.append(t_orig)

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    best_threshold = optimize_threshold(val_preds, val_targets)

    # 2. Predict on Test Set
    print(f"Generating predictions using threshold: {best_threshold:.4f}")
    submission_rows = []

    with torch.no_grad():
        for batch in test_loader:
            images, ids = batch
            images = images.to(device)

            logits = model(images)
            preds = torch.sigmoid(logits)

            preds_np = preds.cpu().numpy()

            for i, p in zip(ids, preds_np):
                p_orig = unpad_image(p[0], Config.ORIG_SIZE)

                # Binarize
                mask_bin = (p_orig > best_threshold).astype(np.uint8)

                # RLE Encode
                rle = rle_encode(mask_bin)
                submission_rows.append([i, rle])

    # Save
    df = pd.DataFrame(submission_rows, columns=["id", "rle_mask"])
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
