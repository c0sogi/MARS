import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from library.utils import calc_map_score
from library.losses import CombinedLoss, StableBCELoss


class AverageMeter:
    """Computes and stores the average and current value."""

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


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using Stabilized Two-Stage Soft-Self-Training logic.

    Logic:
    - Identifies if a sample is Real (Hard GT) or Pseudo (Soft Target).
    - Depth Masking:
        - Real: Replace depth with 0 with p=0.5 (Bernoulli Masking).
        - Pseudo: Force depth to 0.
    - Loss:
        - Real: CombinedLoss (Lovasz + BCE).
        - Pseudo: StableBCELoss (Soft BCE).
    """
    model.train()
    losses = AverageMeter()

    # Instantiate losses locally to ensure compliance with the strategy
    criterion_supervised = CombinedLoss().to(device)
    criterion_pseudo = StableBCELoss().to(device)

    # Progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, (images, masks, depths, ids) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        batch_size = images.size(0)

        # 1. Identify Real vs Pseudo samples
        # Heuristic: If mask contains values strictly between 0 and 1 (exclusive of tolerance), it's soft/pseudo.
        # GT masks are 0.0 or 1.0.
        with torch.no_grad():
            # Check per sample
            # Reshape to (B, -1)
            masks_flat = masks.view(batch_size, -1)
            # Check for values in (0.001, 0.999)
            is_soft = (masks_flat > 1e-3) & (masks_flat < 1 - 1e-3)
            is_pseudo = is_soft.any(dim=1)  # (B,) boolean

        # 2. Depth Masking
        # Create a copy of depths to modify
        z_input = depths.clone()

        # Case A: Pseudo samples -> Force Depth 0
        if is_pseudo.any():
            z_input[is_pseudo] = 0.0

        # Case B: Real samples -> Bernoulli Masking (p=0.5)
        if (~is_pseudo).any():
            real_indices = torch.where(~is_pseudo)[0]
            # Generate random mask: 1 means replace with 0
            dropout_mask = torch.rand(len(real_indices), device=device) < 0.5
            indices_to_zero = real_indices[dropout_mask]
            z_input[indices_to_zero] = 0.0

        # 3. Forward Pass
        optimizer.zero_grad()
        logits = model(images, z_input)

        # 4. Loss Calculation
        total_loss = 0.0

        # Supervised Loss for Real Samples
        if (~is_pseudo).any():
            real_logits = logits[~is_pseudo]
            real_targets = masks[~is_pseudo]
            loss_sup = criterion_supervised(real_logits, real_targets)
            # Weighted by number of samples
            total_loss += loss_sup * (~is_pseudo).sum()

        # Distillation Loss for Pseudo Samples
        if is_pseudo.any():
            pseudo_logits = logits[is_pseudo]
            pseudo_targets = masks[is_pseudo]
            loss_unsup = criterion_pseudo(pseudo_logits, pseudo_targets)
            # Weighted by number of samples
            total_loss += loss_unsup * is_pseudo.sum()

        # Normalize by batch size
        final_loss = total_loss / batch_size

        # 5. Backward
        final_loss.backward()
        optimizer.step()

        losses.update(final_loss.item(), batch_size)
        pbar.set_postfix(loss=losses.avg)

    return losses.avg


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Computes Loss (Combined) and mAP.
    """
    model.eval()
    losses = AverageMeter()
    map_scores = []

    criterion = CombinedLoss().to(device)

    with torch.no_grad():
        pbar = tqdm(loader, desc="[Eval]", leave=False)
        for images, masks, depths, ids in pbar:
            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            # Validation uses true depths (no masking)
            logits = model(images, depths)

            # Loss
            loss = criterion(logits, masks)
            losses.update(loss.item(), images.size(0))

            # mAP Calculation
            # Convert logits to probabilities
            probs = torch.sigmoid(logits)
            batch_score = calc_map_score(probs, masks)
            map_scores.append(batch_score)

    mean_map = np.mean(map_scores)
    print(f"Validation Loss: {losses.avg:.8f}")
    print(f"Validation mAP: {mean_map:.8f}")

    return losses.avg, mean_map


def predict_proba(model, loader, device):
    """
    Generates soft probability maps for the dataset.
    Used for Test Inference and Pseudo-Label generation.

    Features:
    - Forces Depth = 0 (Robust Mode).
    - TTA: Horizontal Flip.
    """
    model.eval()
    preds_dict = {}

    with torch.no_grad():
        pbar = tqdm(loader, desc="[Predict]", leave=False)
        for images, _, _, ids in pbar:
            images = images.to(device)

            # Force Depth = 0 for all test inference
            # Depth tensor shape (B, 1)
            z_zero = torch.zeros(
                (images.size(0), 1), device=device, dtype=torch.float32
            )

            # TTA: Original
            logits_orig = model(images, z_zero)
            probs_orig = torch.sigmoid(logits_orig)

            # TTA: Horizontal Flip
            images_flipped = torch.flip(images, dims=[3])  # [B, C, H, W], flip W
            logits_flipped = model(images_flipped, z_zero)
            probs_flipped = torch.sigmoid(logits_flipped)
            probs_flipped = torch.flip(probs_flipped, dims=[3])  # Flip back

            # Average
            probs_avg = (probs_orig + probs_flipped) / 2.0

            # Store results
            # Convert to numpy
            probs_np = probs_avg.cpu().numpy()  # (B, 1, 128, 128)

            for i, img_id in enumerate(ids):
                # Remove channel dim: (128, 128)
                pred_map = probs_np[i, 0, :, :]
                preds_dict[img_id] = pred_map

    return preds_dict
