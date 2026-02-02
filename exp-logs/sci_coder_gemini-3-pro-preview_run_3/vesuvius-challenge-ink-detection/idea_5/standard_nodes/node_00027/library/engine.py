import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import gc
from collections import defaultdict

from library.config import Config
from library.utils import fbeta_score, get_optimal_threshold, rle_encode


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: logits (B, 1, H, W)
        # targets: binary (B, 1, H, W)

        preds = torch.sigmoid(preds)

        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


def train_one_epoch(model, loader, optimizer, device):
    """
    Training loop for one epoch.
    """
    model.train()

    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = DiceLoss()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (volumes, labels) in enumerate(loader):
        volumes = volumes.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        outputs = model(volumes)

        loss_bce = bce_fn(outputs, labels)
        loss_dice = dice_fn(outputs, labels)

        loss = (Config.BCE_WEIGHT * loss_bce) + (Config.DICE_WEIGHT * loss_dice)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def apply_tta(model, inputs):
    """
    Applies Test-Time Augmentation (8 views: 4 rotations * 2 flips).
    """
    # List of augmentations: (k_rot, flip_dim)
    # k_rot: 0..3 (90 degree steps)
    # flip_dim: None or [3] (horizontal flip, applied before rotation)

    transforms = [
        (0, None),
        (1, None),
        (2, None),
        (3, None),
        (0, [3]),
        (1, [3]),
        (2, [3]),
        (3, [3]),
    ]

    accumulated_preds = None

    for k, flip_dims in transforms:
        x = inputs.clone()

        # Apply Augmentation
        if flip_dims is not None:
            x = torch.flip(x, flip_dims)
        if k > 0:
            x = torch.rot90(x, k, [2, 3])

        # Predict
        with torch.no_grad():
            y = model(x)
            y = torch.sigmoid(y)  # Convert logits to prob

        # Revert Augmentation on Output
        if k > 0:
            y = torch.rot90(y, -k, [2, 3])
        if flip_dims is not None:
            y = torch.flip(y, flip_dims)

        if accumulated_preds is None:
            accumulated_preds = y
        else:
            accumulated_preds += y

    # Average
    return accumulated_preds / len(transforms)


def reconstruct_fragments(model, loader, device, tta=False):
    """
    Runs inference on the loader and reconstructs full fragment maps.
    Handles overlapping tiles by averaging.
    """
    model.eval()

    # Access dataset to get dimensions for each fragment
    dataset = loader.dataset
    data_map = dataset.data_map

    # buffers for accumulation
    # preds_map: {frag_id: np.array(H, W)}
    # count_map: {frag_id: np.array(H, W)}
    preds_map = {}
    count_map = {}

    # Initialize buffers
    for frag_id, info in data_map.items():
        h, w = info["orig_h"], info["orig_w"]
        preds_map[frag_id] = np.zeros((h, w), dtype=np.float32)
        count_map[frag_id] = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for batch in loader:
            # Unpack batch
            # If val: volumes, labels, meta
            # If test: volumes, meta
            if len(batch) == 3:
                volumes, _, meta = batch
            else:
                volumes, meta = batch

            volumes = volumes.to(device, dtype=torch.float32)

            if tta:
                probs = apply_tta(model, volumes)
            else:
                logits = model(volumes)
                probs = torch.sigmoid(logits)

            probs = probs.cpu().numpy()

            # Scatter back to full map
            # meta is a dict of lists (batch items)
            batch_size = volumes.shape[0]
            for i in range(batch_size):
                frag_id = meta["fragment_id"][i]
                y = int(meta["y"][i])
                x = int(meta["x"][i])
                h_crop = int(meta["h"][i])
                w_crop = int(meta["w"][i])

                # Extract prediction for this item (1, H, W) -> (H, W)
                pred_patch = probs[i, 0, :, :]

                # Accumulate
                preds_map[frag_id][y : y + h_crop, x : x + w_crop] += pred_patch
                count_map[frag_id][y : y + h_crop, x : x + w_crop] += 1.0

    # Normalize by counts
    for frag_id in preds_map:
        # Avoid division by zero (should not happen if grid covers everything)
        mask = count_map[frag_id] > 0
        preds_map[frag_id][mask] /= count_map[frag_id][mask]

    return preds_map


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns average F0.5 score and the optimal threshold found.
    """
    print("Evaluating on Validation Set...")

    # Reconstruct full probability maps
    preds_map = reconstruct_fragments(model, loader, device, tta=Config.TTA_ENABLED)

    # Calculate metrics per fragment and average
    f05_scores = []
    dataset = loader.dataset

    # We will concatenate all pixels from all validation fragments to find a global threshold
    # This is more robust than averaging thresholds per fragment
    all_preds = []
    all_targets = []

    for frag_id, pred_img in preds_map.items():
        # Get Ground Truth
        # Note: dataset.data_map[frag_id]["label"] is a numpy array (0/1)
        target_img = dataset.data_map[frag_id]["label"]
        mask_img = dataset.data_map[frag_id]["mask"]

        if target_img is None:
            continue

        # Only evaluate on valid mask area
        valid_mask = mask_img > 0

        flat_preds = pred_img[valid_mask]
        flat_targets = target_img[valid_mask]

        all_preds.append(flat_preds)
        all_targets.append(flat_targets)

    if not all_preds:
        return 0.0, 0.5

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Find optimal threshold
    best_thresh, best_score = get_optimal_threshold(all_preds, all_targets)

    print(f"Validation F0.5 Score: {best_score}")
    print(f"Optimal Threshold: {best_thresh}")

    return best_score, best_thresh


def predict_and_submit(model, loader, device, threshold):
    """
    Generates predictions for the test set and writes submission.csv.
    """
    print("Generating Predictions for Test Set...")

    # Reconstruct full maps
    preds_map = reconstruct_fragments(model, loader, device, tta=Config.TTA_ENABLED)

    submission_data = []

    for frag_id, pred_img in preds_map.items():
        # Apply mask to ensure we don't predict outside valid area
        # (Though model shouldn't anyway, good for safety)
        mask_img = loader.dataset.data_map[frag_id]["mask"]

        # Binarize
        binary_pred = (pred_img > threshold).astype(np.uint8)

        # Mask out invalid regions
        if mask_img is not None:
            binary_pred = binary_pred * (mask_img > 0).astype(np.uint8)

        # RLE Encode
        rle_str = rle_encode(binary_pred)

        submission_data.append({"Id": frag_id, "Predicted": rle_str})

    # Create DataFrame
    df_sub = pd.DataFrame(submission_data)

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
