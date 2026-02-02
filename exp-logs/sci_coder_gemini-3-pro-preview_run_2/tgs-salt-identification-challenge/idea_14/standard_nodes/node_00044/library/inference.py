import torch
import numpy as np
import pandas as pd
import os
from library.utils import rle_encode, calculate_iou_map

# Constants for unpadding (128x128 -> 101x101)
# These must match the padding logic used in the dataset (PadIfNeeded min_height=128, min_width=128)
# For 101x101 input, padding is typically 13 top/left and 14 bottom/right.
PAD_TOP = 13
PAD_LEFT = 13
ORIG_SIZE = 101


def predict_proba(model, dataloader, device, force_zero_depth=True):
    """
    Performs inference on the provided dataloader using the model.

    Features:
    - Test-Time Augmentation (Horizontal Flip).
    - Unpadding (Center Crop from 128x128 to 101x101).
    - Optional forcing of depth input to 0 (for robustness/test set).

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing images.
        device (torch.device): Device to run inference on.
        force_zero_depth (bool): Whether to replace depth inputs with zeros.

    Returns:
        dict: {
            'ids': list of image IDs,
            'predictions': np.ndarray of shape (N, 101, 101) with probabilities,
            'masks': np.ndarray of shape (N, 101, 101) with ground truth (if available)
        }
    """
    model.eval()

    all_probs = []
    all_masks = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch: images, masks, depths, ids
            images, masks, depths, ids = batch

            images = images.to(device)

            # Handle Depth
            if force_zero_depth:
                # Create zero depth tensor (B, 1) to simulate missing metadata / generalist mode
                depth_input = torch.zeros(
                    (images.size(0), 1), dtype=torch.float32, device=device
                )
            else:
                depth_input = depths.to(device)

            # 1. Original Prediction
            logits_orig = model(images, depth_input)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. TTA: Horizontal Flip
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip, depth_input)
            probs_flip = torch.sigmoid(logits_flip)
            probs_flip = torch.flip(probs_flip, dims=[3])  # Flip back

            # Average
            probs_avg = (probs_orig + probs_flip) / 2.0

            # Unpad / Center Crop
            # Crop from 128x128 back to 101x101
            # Indices: 13 : 13+101
            probs_cropped = probs_avg[
                :, :, PAD_TOP : PAD_TOP + ORIG_SIZE, PAD_LEFT : PAD_LEFT + ORIG_SIZE
            ]

            # Convert to numpy
            probs_np = probs_cropped.cpu().numpy()

            # Squeeze channel dim (B, 1, H, W) -> (B, H, W)
            if probs_np.ndim == 4 and probs_np.shape[1] == 1:
                probs_np = probs_np.squeeze(1)

            all_probs.append(probs_np)
            all_ids.extend(ids)

            # Handle Masks (for validation)
            if masks is not None:
                # Masks are also 128x128, need cropping to match prediction
                masks_cropped = masks[
                    :, :, PAD_TOP : PAD_TOP + ORIG_SIZE, PAD_LEFT : PAD_LEFT + ORIG_SIZE
                ]
                masks_np = masks_cropped.cpu().numpy()

                # Squeeze channel dim if present
                if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                    masks_np = masks_np.squeeze(1)

                all_masks.append(masks_np)

    # Concatenate results
    predictions = np.concatenate(all_probs, axis=0)

    masks_arr = None
    if len(all_masks) > 0:
        masks_arr = np.concatenate(all_masks, axis=0)
        # Ensure binary
        masks_arr = (masks_arr > 0).astype(np.uint8)

    return {"ids": all_ids, "predictions": predictions, "masks": masks_arr}


def optimize_threshold(y_true, y_pred_probs):
    """
    Performs a linear search to find the binarization threshold that maximizes
    the competition metric (mAP at IoU thresholds 0.5:0.05:0.95).

    Args:
        y_true (np.ndarray): Ground truth binary masks.
        y_pred_probs (np.ndarray): Predicted probabilities.

    Returns:
        float: The optimal threshold.
    """
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_map = -1.0
    best_thresh = 0.5

    # Ensure y_true is binary
    y_true = (y_true > 0).astype(np.uint8)

    print("Optimizing threshold on validation set...")
    for t in thresholds:
        y_pred_bin = (y_pred_probs > t).astype(np.uint8)
        score = calculate_iou_map(y_true, y_pred_bin)

        if score > best_map:
            best_map = score
            best_thresh = t

    print(f"Best Threshold: {best_thresh} with mAP: {best_map}")
    return best_thresh


def generate_submission_csv(
    ids, y_pred_probs, threshold, output_path="./submission/submission.csv"
):
    """
    Generates a submission CSV file from predicted probabilities.

    Args:
        ids (list): List of image IDs.
        y_pred_probs (np.ndarray): Predicted probabilities.
        threshold (float): Threshold for binarization.
        output_path (str): Path to save the CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Binarize
    y_pred_bin = (y_pred_probs > threshold).astype(np.uint8)

    rle_list = []
    for i in range(len(ids)):
        mask = y_pred_bin[i]
        rle = rle_encode(mask)
        rle_list.append(rle)

    df = pd.DataFrame({"id": ids, "rle_mask": rle_list})

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
