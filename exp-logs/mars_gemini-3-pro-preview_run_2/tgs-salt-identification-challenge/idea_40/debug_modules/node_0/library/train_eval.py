import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import calc_map_score, unpad_image, rle_encode
from library.losses import CombinedLoss, AuxiliaryMSELoss, StableBCELoss


def train_teacher_epoch(model, loader, optimizer, device, loss_fn):
    """
    Trains the Specialist Teacher for one epoch using strictly FP32.

    Args:
        model (nn.Module): The teacher model (SaltNet in 'teacher' mode).
        loader (DataLoader): DataLoader for labeled training data.
        optimizer (Optimizer): Torch optimizer.
        device (str): 'cuda' or 'cpu'.
        loss_fn (nn.Module): Loss function (CombinedLoss).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    # Progress bar for monitoring
    pbar = tqdm(loader, desc="Train Teacher", leave=False, dynamic_ncols=True)

    for images, masks, depths, _ in pbar:
        # Enforce FP32 for stability
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        depths = depths.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # Teacher forward pass requires explicit depth injection
        logits = model(images, depth=depths)

        loss = loss_fn(logits, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)


def train_student_epoch(
    model,
    labeled_loader,
    unlabeled_loader,
    optimizer,
    device,
    seg_loss_fn,
    aux_loss_fn,
    soft_loss_fn,
):
    """
    Trains the Generalist Student for one epoch using Multi-Task Distillation.
    Combines labeled data (Seg + Depth Aux) and unlabeled data (Soft Distillation).

    Args:
        model (nn.Module): The student model (SaltNet in 'student' mode).
        labeled_loader (DataLoader): Loader for labeled data (GT masks + depths).
        unlabeled_loader (DataLoader): Loader for unlabeled data (Soft masks).
        optimizer (Optimizer): Torch optimizer.
        device (str): 'cuda' or 'cpu'.
        seg_loss_fn (nn.Module): CombinedLoss for labeled segmentation.
        aux_loss_fn (nn.Module): AuxiliaryMSELoss for depth regression.
        soft_loss_fn (nn.Module): StableBCELoss for soft target distillation.

    Returns:
        dict: Dictionary containing average losses (total, seg, aux, soft).
    """
    model.train()

    running_loss_total = 0.0
    running_loss_seg = 0.0
    running_loss_aux = 0.0
    running_loss_soft = 0.0

    # Iterator for unlabeled data to handle length mismatch
    unlabeled_iter = iter(unlabeled_loader)

    pbar = tqdm(labeled_loader, desc="Train Student", leave=False, dynamic_ncols=True)

    for images_l, masks_l, depths_l, _ in pbar:
        # ---------------------------
        # 1. Prepare Data (FP32)
        # ---------------------------
        images_l = images_l.to(device, dtype=torch.float32)
        masks_l = masks_l.to(device, dtype=torch.float32)
        depths_l = depths_l.to(device, dtype=torch.float32)

        # Fetch unlabeled batch (cycle if necessary)
        try:
            images_u, soft_masks_u, _, _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            images_u, soft_masks_u, _, _ = next(unlabeled_iter)

        images_u = images_u.to(device, dtype=torch.float32)
        soft_masks_u = soft_masks_u.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # ---------------------------
        # 2. Labeled Forward (Multi-Task)
        # ---------------------------
        # Student returns (logits, aux_depth)
        logits_l, aux_depth_l = model(images_l)

        loss_seg = seg_loss_fn(logits_l, masks_l)
        loss_aux = aux_loss_fn(aux_depth_l, depths_l)

        # ---------------------------
        # 3. Unlabeled Forward (Distillation)
        # ---------------------------
        # We ignore aux output for unlabeled data as we lack GT depth
        logits_u, _ = model(images_u)

        loss_soft = soft_loss_fn(logits_u, soft_masks_u)

        # ---------------------------
        # 4. Optimization
        # ---------------------------
        total_loss = loss_seg + loss_aux + loss_soft

        total_loss.backward()
        optimizer.step()

        # Logging
        running_loss_total += total_loss.item()
        running_loss_seg += loss_seg.item()
        running_loss_aux += loss_aux.item()
        running_loss_soft += loss_soft.item()

        pbar.set_postfix(total_loss=f"{total_loss.item():.4f}")

    num_batches = len(labeled_loader)
    return {
        "loss_total": running_loss_total / num_batches,
        "loss_seg": running_loss_seg / num_batches,
        "loss_aux": running_loss_aux / num_batches,
        "loss_soft": running_loss_soft / num_batches,
    }


def validate(model, loader, device, loss_fn):
    """
    Evaluates the model on the validation set.
    Calculates average loss and mean Average Precision (mAP).

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation DataLoader.
        device (str): 'cuda' or 'cpu'.
        loss_fn (nn.Module): Loss function.

    Returns:
        tuple: (avg_loss, avg_map)
    """
    model.eval()
    running_loss = 0.0
    running_map = 0.0

    with torch.no_grad():
        for images, masks, depths, _ in tqdm(
            loader, desc="Validation", leave=False, dynamic_ncols=True
        ):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Handle mode-specific inputs
            if model.mode == "teacher":
                depths = depths.to(device, dtype=torch.float32)
                logits = model(images, depth=depths)
            else:
                # Student mode returns (logits, aux), we only need logits for validation
                logits, _ = model(images)

            loss = loss_fn(logits, masks)
            running_loss += loss.item()

            # Calculate mAP
            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Binarize at 0.5 for monitoring metric (mAP function sweeps IoU thresholds)
            pred_bin = (probs > 0.5).float().cpu().numpy().astype(np.uint8)
            true_masks = masks.cpu().numpy().astype(np.uint8)

            # calc_map_score calculates mean AP for the batch
            batch_map = calc_map_score(pred_bin, true_masks)
            running_map += batch_map

    return running_loss / len(loader), running_map / len(loader)


def generate_submission(model, test_loader, device, threshold=0.5):
    """
    Generates predictions for the test set and saves to submission.csv.
    Applies Test-Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): Trained model (Student).
        test_loader (DataLoader): Test dataset loader.
        device (str): Device.
        threshold (float): Binarization threshold.
    """
    model.eval()
    submission_data = []

    print(f"Generating submission with threshold {threshold}...")

    with torch.no_grad():
        for images, _, _, ids in tqdm(
            test_loader, desc="Inference", dynamic_ncols=True
        ):
            images = images.to(device, dtype=torch.float32)

            # 1. Forward Pass (Original)
            if model.mode == "teacher":
                # Teacher requires depth, but test set depth injection is not part of this strategy
                # This function assumes Student model usage.
                logits, _ = model(images)
            else:
                logits, _ = model(images)

            probs = torch.sigmoid(logits)

            # 2. Forward Pass (Horizontal Flip TTA)
            images_flipped = torch.flip(images, dims=[3])

            if model.mode == "teacher":
                logits_flipped, _ = model(images_flipped)
            else:
                logits_flipped, _ = model(images_flipped)

            probs_flipped = torch.sigmoid(logits_flipped)
            probs_flipped_back = torch.flip(probs_flipped, dims=[3])

            # 3. Average Predictions
            avg_probs = (probs + probs_flipped_back) / 2.0
            avg_probs_np = avg_probs.cpu().numpy()

            # 4. Process Batch
            for i in range(len(ids)):
                img_id = ids[i]
                # Extract single image probability map (H, W)
                prob_map = avg_probs_np[i, 0, :, :]

                # Unpad (128x128 -> 101x101)
                prob_map_cropped = unpad_image(prob_map)

                # Binarize
                mask_bin = (prob_map_cropped > threshold).astype(np.uint8)

                # RLE Encode
                rle = rle_encode(mask_bin)
                submission_data.append({"id": img_id, "rle_mask": rle})

    # Save to CSV
    sub_df = pd.DataFrame(submission_data)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
