import os
import torch
import numpy as np
import time
from library import config, utils, losses

# =============================================================================
# Core Training Functions
# =============================================================================


def train_teacher_epoch(model, loader, optimizer, loss_fn, device):
    """
    Trains the Privileged Teacher for one epoch using Ground Truth Depth.
    """
    model.train()
    running_loss = 0.0

    for images, masks, depths, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()

        # Teacher Forward: Uses Image + Depth
        logits = model(images, depths)

        # Loss Calculation
        loss = loss_fn(logits, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def train_student_epoch(student, teacher, loader, optimizer, loss_fn, device):
    """
    Trains the Multi-Task Student via Distillation.
    Teacher is frozen and provides soft targets.
    Student predicts Mask + Aux Depth.
    """
    student.train()
    teacher.eval()

    running_loss = 0.0
    running_components = {"loss_seg": 0.0, "loss_distill": 0.0, "loss_depth": 0.0}

    for images, masks, depths, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        # Get Teacher Soft Targets (No Grad)
        with torch.no_grad():
            teacher_logits = teacher(images, depths)

        optimizer.zero_grad()

        # Student Forward: Image Only
        student_logits, student_depth_pred = student(images)

        # Composite Loss
        loss, components = loss_fn(
            student_logits, teacher_logits, student_depth_pred, masks, depths
        )

        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        for k, v in components.items():
            running_components[k] += v * bs

    dataset_size = len(loader.dataset)
    avg_loss = running_loss / dataset_size
    avg_components = {k: v / dataset_size for k, v in running_components.items()}

    return avg_loss, avg_components


# =============================================================================
# Evaluation & Optimization
# =============================================================================


def validate(model, loader, device, is_teacher=False, binarization_threshold=0.5):
    """
    Evaluates the model on the validation set.
    Handles unpadding of predictions to original 101x101 size before metric calculation.
    """
    model.eval()
    all_preds = []
    all_masks = []

    # Calculate unpadding indices for center crop
    # 128 -> 101: diff is 27. Pad top=13, left=13.
    pad_top = (config.IMG_SIZE - config.ORIG_SIZE) // 2
    pad_left = (config.IMG_SIZE - config.ORIG_SIZE) // 2
    h_end = pad_top + config.ORIG_SIZE
    w_end = pad_left + config.ORIG_SIZE

    with torch.no_grad():
        for images, masks, depths, _ in loader:
            images = images.to(device)

            # Forward pass
            if is_teacher:
                depths = depths.to(device)
                logits = model(images, depths)
            else:
                # Student returns (mask, depth), we only need mask
                logits, _ = model(images)

            probs = torch.sigmoid(logits)

            # Unpad predictions and masks to original size (101x101)
            # Shapes: (B, 1, 128, 128) -> (B, 1, 101, 101)
            probs_cropped = probs[:, :, pad_top:h_end, pad_left:w_end]
            masks_cropped = masks[:, :, pad_top:h_end, pad_left:w_end]

            # Binarize based on provided threshold
            # We convert to float 0.0/1.0 so utils.calc_iou_batch (which uses >0.5) works correctly
            preds_bin = (probs_cropped > binarization_threshold).float()

            all_preds.append(preds_bin.cpu())
            all_masks.append(masks_cropped.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Calculate mAP
    # Note: utils.calc_map_score expects inputs where >0.5 indicates positive
    score = utils.calc_map_score(all_preds, all_masks)

    return score


def optimize_threshold(model, loader, device, is_teacher=False):
    """
    Performs a linear search to find the optimal binarization threshold.
    """
    thresholds = np.arange(0.3, 0.75, 0.05)
    best_score = -1.0
    best_threshold = 0.5

    print("Optimizing binarization threshold...")

    # To save time, pre-compute unpadded probabilities once
    model.eval()
    all_probs = []
    all_masks = []

    pad_top = (config.IMG_SIZE - config.ORIG_SIZE) // 2
    pad_left = (config.IMG_SIZE - config.ORIG_SIZE) // 2
    h_end = pad_top + config.ORIG_SIZE
    w_end = pad_left + config.ORIG_SIZE

    with torch.no_grad():
        for images, masks, depths, _ in loader:
            images = images.to(device)
            if is_teacher:
                depths = depths.to(device)
                logits = model(images, depths)
            else:
                logits, _ = model(images)

            probs = torch.sigmoid(logits)

            # Unpad
            probs_cropped = probs[:, :, pad_top:h_end, pad_left:w_end].cpu()
            masks_cropped = masks[:, :, pad_top:h_end, pad_left:w_end].cpu()

            all_probs.append(probs_cropped)
            all_masks.append(masks_cropped)

    all_probs = torch.cat(all_probs, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Sweep
    for t in thresholds:
        preds_bin = (all_probs > t).float()
        score = utils.calc_map_score(preds_bin, all_masks)
        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"Optimal Threshold: {best_threshold:.2f} with mAP: {best_score}")
    return best_threshold


# =============================================================================
# Training Loops (Fit)
# =============================================================================


def fit_teacher(
    model, train_loader, val_loader, optimizer, loss_fn, device, epochs, save_path
):
    """
    Main loop for Teacher training with Early Stopping.
    """
    best_score = 0.0
    patience = 10
    patience_counter = 0

    print(f"Starting Teacher Training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_teacher_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_score = validate(model, val_loader, device, is_teacher=True)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val mAP: {val_score} | "  # Printing full precision
            f"Time: {elapsed:.0f}s"
        )

        # Checkpoint
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New Best Teacher! Saved to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Teacher training complete. Best mAP: {best_score}")


def fit_student(
    student,
    teacher,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    epochs,
    save_path,
):
    """
    Main loop for Student training with Early Stopping.
    """
    best_score = 0.0
    patience = 10
    patience_counter = 0

    # Determine optimal threshold from teacher to guide expectations (optional, but good practice)
    # We use fixed 0.5 for validation during training for stability

    print(f"Starting Student Training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        loss, components = train_student_epoch(
            student, teacher, train_loader, optimizer, loss_fn, device
        )
        val_score = validate(student, val_loader, device, is_teacher=False)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {loss:.4f} (Seg: {components['loss_seg']:.4f}, Distill: {components['loss_distill']:.4f}, Depth: {components['loss_depth']:.4f}) | "
            f"Val mAP: {val_score} | "
            f"Time: {elapsed:.0f}s"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(student.state_dict(), save_path)
            print(f"  >>> New Best Student! Saved to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Student training complete. Best mAP: {best_score}")
