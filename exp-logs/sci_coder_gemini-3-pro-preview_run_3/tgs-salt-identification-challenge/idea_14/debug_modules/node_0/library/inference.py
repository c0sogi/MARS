import os
import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
from library.config import Config
from library.utils import rle_encode, calculate_iou_batch


def predict_with_tta(model, loader, device):
    """
    Performs inference with Test-Time Augmentation (Horizontal Flip).
    Handles both Validation (Image, Mask) and Test (Image, ID) loaders.
    Automatically crops predictions from padded size (128x128) to original size (101x101).

    Args:
        model: PyTorch model.
        loader: DataLoader.
        device: Device string or torch.device.

    Returns:
        dict: {
            'preds': np.ndarray (N, H, W) float32 [0, 1],
            'targets': np.ndarray (N, H, W) uint8 (Optional, if masks present),
            'ids': list of strings (Optional, if IDs present)
        }
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    has_masks = False
    has_ids = False

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)

            # Determine batch content type
            # Validation loader returns [images, masks]
            # Test loader returns [images, ids]
            if isinstance(batch[1], torch.Tensor):
                masks = batch[1]
                has_masks = True
                # Move to CPU immediately to save GPU memory
                all_targets.append(masks.cpu().numpy())
            else:
                ids = batch[1]
                has_ids = True
                all_ids.extend(ids)

            # TTA Pass 1: Original
            with amp.autocast():
                logits_orig = model(images)
                probs_orig = torch.sigmoid(logits_orig)

                # TTA Pass 2: Horizontal Flip
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
                probs_flip = torch.sigmoid(logits_flip)

            # Flip predictions back
            probs_flip_back = torch.flip(probs_flip, dims=[3])

            # Average predictions
            probs_avg = (probs_orig + probs_flip_back) / 2.0

            all_preds.append(probs_avg.cpu().numpy())

    # Concatenate batches
    # Shape: (N, 1, 128, 128)
    preds_arr = np.concatenate(all_preds, axis=0)

    # Remove channel dimension
    if preds_arr.ndim == 4 and preds_arr.shape[1] == 1:
        preds_arr = preds_arr[:, 0, :, :]

    # Center Crop to Original Size (101x101)
    if preds_arr.shape[-1] != Config.ORIG_SIZE:
        h, w = preds_arr.shape[-2:]
        start_h = (h - Config.ORIG_SIZE) // 2
        start_w = (w - Config.ORIG_SIZE) // 2
        preds_arr = preds_arr[
            :,
            start_h : start_h + Config.ORIG_SIZE,
            start_w : start_w + Config.ORIG_SIZE,
        ]

    result = {"preds": preds_arr}

    if has_masks:
        targets_arr = np.concatenate(all_targets, axis=0)
        # Remove channel dimension
        if targets_arr.ndim == 4 and targets_arr.shape[1] == 1:
            targets_arr = targets_arr[:, 0, :, :]

        # Crop targets
        if targets_arr.shape[-1] != Config.ORIG_SIZE:
            h, w = targets_arr.shape[-2:]
            start_h = (h - Config.ORIG_SIZE) // 2
            start_w = (w - Config.ORIG_SIZE) // 2
            targets_arr = targets_arr[
                :,
                start_h : start_h + Config.ORIG_SIZE,
                start_w : start_w + Config.ORIG_SIZE,
            ]

        # Ensure binary uint8
        targets_arr = (targets_arr > 0.5).astype(np.uint8)
        result["targets"] = targets_arr

    if has_ids:
        result["ids"] = all_ids

    return result


def optimize_threshold(preds, targets):
    """
    Finds the global binarization threshold that maximizes the competition metric (mAP).

    Args:
        preds: np.ndarray (N, H, W) probabilities.
        targets: np.ndarray (N, H, W) binary masks.

    Returns:
        float: Best threshold.
    """
    # Sweep range: 0.3 to 0.75
    thresholds = np.arange(0.3, 0.76, 0.05)
    best_t = 0.5
    best_score = -1.0

    print("Optimizing binarization threshold...")
    for t in thresholds:
        # calculate_iou_batch computes mAP over IoU thresholds 0.5:0.95
        score = calculate_iou_batch(preds, targets, threshold=t)
        print(f"Threshold {t:.2f} | mAP: {score:.8f}")

        if score > best_score:
            best_score = score
            best_t = t

    print(f"Best Threshold Found: {best_t:.2f} with mAP: {best_score:.8f}")
    return best_t


def generate_submission(preds, ids, threshold, output_path):
    """
    Generates the submission CSV file using RLE encoding.

    Args:
        preds: np.ndarray (N, H, W) probabilities.
        ids: list of strings.
        threshold: float, binarization threshold.
        output_path: str, path to save CSV.
    """
    print(f"Generating submission with threshold {threshold:.4f}...")

    # Binarize predictions
    preds_bin = (preds > threshold).astype(np.uint8)

    rle_list = []
    # Iterate and encode
    for i in range(len(ids)):
        rle = rle_encode(preds_bin[i])
        rle_list.append(rle)

    # Create DataFrame
    df = pd.DataFrame({"id": ids, "rle_mask": rle_list})

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
