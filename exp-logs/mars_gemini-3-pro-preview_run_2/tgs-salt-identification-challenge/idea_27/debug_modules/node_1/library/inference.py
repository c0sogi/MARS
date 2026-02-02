import os
import numpy as np
import torch
import pandas as pd
from library.utils import rle_encode, calculate_map


def predict_with_tta(model, loader, device):
    """
    Performs inference with Test-Time Augmentation (Horizontal Flip).
    Averages predictions from original and flipped images.
    Crops the 128x128 model output back to the original 101x101 size.

    Args:
        model: PyTorch model.
        loader: DataLoader.
        device: torch.device.

    Returns:
        tuple: (predictions_np, masks_np, ids_list)
               predictions_np: (N, 101, 101) float array of probabilities.
               masks_np: (N, 101, 101) uint8 array of ground truth (or None).
               ids_list: List of image IDs.
    """
    model.eval()
    all_preds = []
    all_masks = []
    all_ids = []

    # Crop indices for 128x128 -> 101x101
    # Padding (128-101=27) is split: 13 top/left, 14 bottom/right
    start_idx = 13
    end_idx = 13 + 101  # 114

    with torch.no_grad():
        for batch in loader:
            # Unpack batch (handle test vs val/train structure)
            if len(batch) == 4:
                images, masks, depths, ids = batch
                has_masks = True
            elif len(batch) == 3:
                images, depths, ids = batch
                has_masks = False
                masks = None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            images = images.to(device)

            # 1. Forward Pass (Original)
            logits, _ = model(images)
            probs = torch.sigmoid(logits)
            if probs.dim() == 4:
                probs = probs.squeeze(1)  # (B, H, W)

            # 2. Forward Pass (Horizontal Flip)
            # Image: (B, C, H, W) -> Flip on dim 3 (Width)
            images_flip = torch.flip(images, dims=[3])
            logits_flip, _ = model(images_flip)
            probs_flip = torch.sigmoid(logits_flip)
            if probs_flip.dim() == 4:
                probs_flip = probs_flip.squeeze(1)

            # Flip probabilities back
            # Probs: (B, H, W) -> Flip on dim 2 (Width)
            probs_flip_back = torch.flip(probs_flip, dims=[2])

            # Average predictions
            avg_probs = (probs + probs_flip_back) / 2.0

            # Center Crop to 101x101
            cropped_probs = avg_probs[:, start_idx:end_idx, start_idx:end_idx]

            # Store results
            all_preds.append(cropped_probs.cpu().numpy())
            all_ids.extend(ids)

            if has_masks:
                all_masks.append(masks.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    if all_masks:
        all_masks = np.concatenate(all_masks, axis=0)
    else:
        all_masks = None

    return all_preds, all_masks, all_ids


def optimize_threshold(model, val_loader, device):
    """
    Finds the optimal binarization threshold by sweeping values on the validation set.

    Args:
        model: PyTorch model.
        val_loader: Validation DataLoader.
        device: torch.device.

    Returns:
        float: The optimal threshold.
    """
    print("Optimizing threshold on validation set...")

    # Get probabilities using TTA
    preds, gts, _ = predict_with_tta(model, val_loader, device)

    # Define search space
    thresholds = np.linspace(0.3, 0.7, 41)  # 0.30, 0.31, ..., 0.70
    best_score = -1.0
    best_thresh = 0.5

    for t in thresholds:
        # Binarize predictions at current threshold
        # calculate_map expects binary inputs (0s and 1s) or probabilities
        # If we pass 0.0/1.0 floats, calculate_map thresholds at 0.5, effectively preserving our binary mask
        binary_preds = (preds > t).astype(np.float32)

        score = calculate_map(binary_preds, gts)

        # Print full precision as requested
        print(f"Threshold: {t:.2f}, mAP: {score}")

        if score > best_score:
            best_score = score
            best_thresh = t

    print(f"Best Threshold: {best_thresh} with mAP: {best_score}")
    return best_thresh


def generate_submission(
    model, test_loader, device, threshold, output_dir="./submission"
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: PyTorch model.
        test_loader: Test DataLoader.
        device: torch.device.
        threshold: Binarization threshold.
        output_dir: Directory to save the submission file.
    """
    print(f"Generating submission with threshold {threshold}...")

    # Get predictions
    preds, _, ids = predict_with_tta(model, test_loader, device)

    # Binarize
    binary_preds = (preds > threshold).astype(np.uint8)

    rle_list = []
    # Iterate and encode
    for i in range(len(ids)):
        mask = binary_preds[i]
        rle = rle_encode(mask)
        rle_list.append(rle)

    # Create DataFrame
    df = pd.DataFrame({"id": ids, "rle_mask": rle_list})

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "submission.csv")
    df.to_csv(out_path, index=False)

    print(f"Submission saved to {out_path}")
