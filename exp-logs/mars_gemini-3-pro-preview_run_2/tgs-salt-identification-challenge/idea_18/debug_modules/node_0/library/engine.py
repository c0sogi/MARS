import torch
import torch.nn as nn
import numpy as np
import itertools
from library.utils import do_kaggle_metric, unpad_image, pad_image


def train_ict_epoch(model, loader, optimizer, criterion, device):
    """
    Phase 1: Internal Consistency Training.
    Trains with two passes per batch:
    1. Specialist Pass: Input Image + True Depth.
    2. Generalist Pass: Input Image + Zero Depth (Mean).
    Accumulates loss from both to enforce robustness.
    """
    model.train()
    losses = []

    for batch in loader:
        # Unpack batch: images, masks, depths, ids
        if len(batch) == 4:
            images, masks, depths, _ = batch
        else:
            continue

        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        # --- Pass 1: Specialist (True Depth) ---
        logits_z = model(images, depths)
        loss_z = criterion(logits_z, masks)

        # --- Pass 2: Generalist (Zero Depth) ---
        # Create zero depths (mean value due to normalization)
        zeros = torch.zeros_like(depths).to(device)
        logits_0 = model(images, zeros)
        loss_0 = criterion(logits_0, masks)

        # Combined Loss
        loss = loss_z + loss_0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def train_adapt_epoch(
    model, labeled_loader, unlabeled_loader, pseudo_labels, optimizer, criterion, device
):
    """
    Phase 2: Soft-Adaptation.
    Fine-tunes on Labeled Data (True Depth, GT) and Unlabeled Data (Zero Depth, Soft Pseudo-Labels).
    Uses soft targets for the unlabeled portion to avoid hard thresholding errors.
    """
    model.train()
    losses = []

    # Cycle through unlabeled data to match the length of the labeled dataset
    iter_unlabeled = itertools.cycle(unlabeled_loader)

    for batch_l in labeled_loader:
        # Get next unlabeled batch
        batch_u = next(iter_unlabeled)

        # --- Labeled Data Processing ---
        images_l, masks_l, depths_l, _ = batch_l
        images_l = images_l.to(device)
        masks_l = masks_l.to(device)
        depths_l = depths_l.to(device)

        # Forward Labeled (Specialist Mode)
        logits_l = model(images_l, depths_l)
        loss_l = criterion(logits_l, masks_l)

        # --- Unlabeled Data Processing ---
        # Test loader returns: image, depth, id
        images_u, depths_u, ids_u = batch_u
        images_u = images_u.to(device)

        # Force Zero Depth for Unlabeled (Generalist Mode)
        zeros_u = torch.zeros_like(depths_u).to(device)

        # Prepare Soft Pseudo-Labels
        # pseudo_labels is a dict {id: (101, 101) numpy array}
        batch_pseudo_masks = []
        for pid in ids_u:
            # Retrieve probability map
            p_mask = pseudo_labels.get(pid)
            if p_mask is None:
                # Fallback (should not happen if logic is correct)
                p_mask = np.zeros((101, 101), dtype=np.float32)

            # Pad to 128x128 to match network input
            p_mask_padded = pad_image(p_mask, target_size=128)
            batch_pseudo_masks.append(p_mask_padded)

        # Convert to tensor (B, 1, H, W)
        masks_u = torch.tensor(np.array(batch_pseudo_masks), dtype=torch.float32)
        masks_u = masks_u.unsqueeze(1).to(device)

        # Forward Unlabeled
        logits_u = model(images_u, zeros_u)

        # Calculate Loss on Unlabeled
        # BCEWithLogitsLoss (inside BCELovasz) handles soft targets effectively
        loss_u = criterion(logits_u, masks_u)

        # Total Loss
        loss = loss_l + loss_u

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def predict_proba(model, loader, device, use_tta=True):
    """
    Generates probability maps for the dataset.
    - Uses Test-Time Augmentation (Horizontal Flip).
    - Enforces Zero Depth for all predictions (Generalist Mode).
    - Unpads predictions back to original 101x101 size.

    Returns:
        Dict {id: (101, 101) numpy array of probabilities}
    """
    model.eval()
    preds = {}

    with torch.no_grad():
        for batch in loader:
            # Handle different loader outputs (Test vs Val)
            if len(batch) == 3:
                images, depths, ids = batch
            elif len(batch) == 4:
                images, _, depths, ids = batch
            else:
                continue

            images = images.to(device)
            # Force Zero Depth for Inference
            zeros = torch.zeros_like(depths).to(device)

            # Forward Original
            logits = model(images, zeros)
            probs = torch.sigmoid(logits)

            if use_tta:
                # Forward Flipped
                images_flip = torch.flip(images, dims=[3])  # Flip width
                logits_flip = model(images_flip, zeros)
                probs_flip = torch.sigmoid(logits_flip)
                # Flip back
                probs_flip = torch.flip(probs_flip, dims=[3])
                # Average
                probs = (probs + probs_flip) / 2.0

            # Move to CPU
            probs_np = probs.cpu().numpy()  # (B, 1, 128, 128)

            for i, pid in enumerate(ids):
                # Extract single map
                p_map = probs_np[i, 0]
                # Unpad to original size (101, 101)
                p_map_unpadded = unpad_image(p_map, original_size=101)
                preds[pid] = p_map_unpadded

    return preds


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.
    1. Generates predictions using TTA.
    2. Aligns predictions with unpadded Ground Truth.
    3. Optimizes the binarization threshold for maximum mAP.

    Returns:
        best_score (float): The best mAP achieved.
        best_thresh (float): The threshold that achieved the best score.
    """
    # 1. Generate Predictions
    preds_dict = predict_proba(model, loader, device, use_tta=True)

    y_true = []
    y_pred = []

    # 2. Align with Ground Truth
    # Iterate loader to get GT masks corresponding to IDs
    # Note: Loader yields Padded masks (128x128). We must unpad them for metric calc.
    gt_map = {}
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                _, masks, _, ids = batch
                masks_np = masks.numpy()  # (B, 1, 128, 128)
                for i, pid in enumerate(ids):
                    mask_padded = masks_np[i, 0]
                    mask_unpadded = unpad_image(mask_padded, original_size=101)
                    gt_map[pid] = mask_unpadded

    # Create aligned lists
    for pid, pred_map in preds_dict.items():
        if pid in gt_map:
            y_pred.append(pred_map)
            y_true.append(gt_map[pid])

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # 3. Optimize Threshold
    # Search range: 0.3 to 0.7 (typical sweet spot for IoU metrics)
    thresholds = np.linspace(0.3, 0.7, 21)
    best_score = -1.0
    best_thresh = 0.5

    for t in thresholds:
        score = do_kaggle_metric(y_pred, y_true, threshold=t)
        if score > best_score:
            best_score = score
            best_thresh = t

    return best_score, best_thresh
