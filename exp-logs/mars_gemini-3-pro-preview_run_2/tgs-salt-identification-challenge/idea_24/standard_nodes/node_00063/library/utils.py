import os
import numpy as np
import cv2
import pandas as pd
from library.config import Config


def pad_image(image, target_h=Config.IMG_H, target_w=Config.IMG_W):
    """
    Pads an image to the target dimensions using reflection padding.
    Default target is 128x128.

    Args:
        image (np.ndarray): Input image of shape (H, W) or (H, W, C).
        target_h (int): Target height.
        target_w (int): Target width.

    Returns:
        np.ndarray: Padded image.
    """
    h, w = image.shape[:2]

    if h == target_h and w == target_w:
        return image

    pad_h = target_h - h
    pad_w = target_w - w

    # Handle cases where image might be larger (unlikely for this dataset)
    if pad_h < 0 or pad_w < 0:
        return cv2.resize(image, (target_w, target_h))

    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return cv2.copyMakeBorder(
        image, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REFLECT
    )


def unpad_image(image, original_h=Config.ORIG_H, original_w=Config.ORIG_W):
    """
    Crops the center of the image back to the original dimensions.

    Args:
        image (np.ndarray): Padded image of shape (H, W) or (H, W, C).
        original_h (int): Original height to crop to.
        original_w (int): Original width to crop to.

    Returns:
        np.ndarray: Unpadded (cropped) image.
    """
    h, w = image.shape[:2]

    pad_h = h - original_h
    pad_w = w - original_w

    if pad_h < 0 or pad_w < 0:
        # If image is smaller than original, resize back (fallback)
        return cv2.resize(image, (original_w, original_h))

    pad_top = pad_h // 2
    pad_left = pad_w // 2

    return image[pad_top : pad_top + original_h, pad_left : pad_left + original_w]


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE) string.
    The format is space-delimited pairs of (start, length).
    1-indexed, column-major order.

    Args:
        mask (np.ndarray): Binary mask (0 or 1).

    Returns:
        str: RLE string.
    """
    # Flatten column-wise (Fortran order)
    pixels = mask.flatten(order="F")

    # We prepend and append 0 to detect runs that start at index 0 or end at the last index
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is start of first run, runs[1] is start of next gap (so length is runs[1]-runs[0])
    # Adjust lengths
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(rle_string, shape=(Config.ORIG_H, Config.ORIG_W)):
    """
    Decodes an RLE string back to a binary mask.

    Args:
        rle_string (str): RLE string.
        shape (tuple): Output shape (H, W).

    Returns:
        np.ndarray: Binary mask.
    """
    if pd.isna(rle_string) or rle_string == "":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # 1-indexed to 0-indexed
    starts -= 1
    ends = starts + lengths

    # Create flat array
    total_pixels = shape[0] * shape[1]
    img = np.zeros(total_pixels, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-wise
    return img.reshape(shape, order="F")


def calc_map_score(pred_masks, true_masks, thresholds=None):
    """
    Calculates the Mean Average Precision (mAP) at IoU thresholds.

    Args:
        pred_masks (np.ndarray): Binary predictions (N, H, W).
        true_masks (np.ndarray): Binary ground truth (N, H, W).
        thresholds (list): List of IoU thresholds. Defaults to Config.IOU_THRESHOLDS.

    Returns:
        float: The mAP score.
    """
    if thresholds is None:
        thresholds = Config.IOU_THRESHOLDS

    # Ensure boolean/binary
    pred_masks = pred_masks > 0
    true_masks = true_masks > 0

    # Calculate intersection and union
    # Sum over spatial dimensions (1, 2)
    intersection = (pred_masks & true_masks).sum(axis=(1, 2))
    union = (pred_masks | true_masks).sum(axis=(1, 2))

    # Calculate IoU
    # Handle edge case: if union is 0 (both empty), IoU is 1.0
    # If union is 0 but intersection is 0, it means both are empty -> 1.0
    # If union > 0, standard calc.

    iou = np.zeros(len(pred_masks), dtype=np.float64)

    # Mask for non-empty union
    non_empty = union > 0
    iou[non_empty] = intersection[non_empty] / union[non_empty]

    # Mask for empty union (both empty) -> IoU = 1
    empty_union = union == 0
    iou[empty_union] = 1.0

    # Calculate precision at each threshold
    # For a single image, precision at threshold t is 1 if IoU > t, else 0.
    # We average this over all thresholds.

    # Shape: (N_samples, N_thresholds)
    # Broadcast iou against thresholds
    # iou: (N, 1), thresholds: (1, T)
    iou_expanded = iou[:, None]
    thresh_expanded = np.array(thresholds)[None, :]

    # matches[i, j] is True if iou[i] > thresholds[j]
    matches = iou_expanded > thresh_expanded

    # Average over thresholds for each image -> AP per image
    ap_per_image = np.mean(matches, axis=1)

    # Mean over dataset
    map_score = np.mean(ap_per_image)

    return map_score


def create_submission(ids, pred_masks, output_path=Config.SUBMISSION_PATH):
    """
    Generates a submission CSV file from predicted masks.

    Args:
        ids (list): List of image IDs.
        pred_masks (np.ndarray): Binary predicted masks (N, H, W).
        output_path (str): Path to save the CSV.
    """
    rles = []
    for mask in pred_masks:
        rles.append(rle_encode(mask))

    df = pd.DataFrame({"id": ids, "rle_mask": rles})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def load_data_with_cache(cache_path, compute_fn, load_cached_data=True):
    """
    Generic caching mechanism for deterministic data processing.

    Args:
        cache_path (str): Path to the .npy file.
        compute_fn (callable): Function to compute data if cache misses.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: The loaded or computed data.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=False)
            return data
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # Compute
    data = compute_fn()

    # Save
    np.save(cache_path, data)

    return data
