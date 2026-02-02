import os
import cv2
import numpy as np
import pandas as pd
import torch
from library import config

# =============================================================================
# RLE Encoding/Decoding
# =============================================================================


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 1 - mask, 0 - background.

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten column-wise
    pixels = mask.T.flatten()
    # Pad start and end to detect runs at boundaries
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(rle_str, shape=(101, 101)):
    """
    Decodes an RLE string into a binary mask.

    Args:
        rle_str (str): Space-delimited RLE string.
        shape (tuple): Target shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if pd.isna(rle_str) or rle_str == "":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


# =============================================================================
# Image Processing (Padding/Unpadding)
# =============================================================================


def pad_image(image, target_size=config.IMG_SIZE):
    """
    Pads an image to the target size using reflection padding.
    Assumes input is (H, W) or (H, W, C).

    Args:
        image (np.ndarray): Input image.
        target_size (int): Target height/width (square).

    Returns:
        np.ndarray: Padded image.
    """
    h, w = image.shape[:2]
    diff_h = target_size - h
    diff_w = target_size - w

    pad_top = diff_h // 2
    pad_bottom = diff_h - pad_top
    pad_left = diff_w // 2
    pad_right = diff_w - pad_left

    # Use BORDER_REFLECT_101 for mirror padding which is standard for this task
    padded = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )

    # If image was (H, W, 1), cv2 might drop the channel dim, restore it
    if len(image.shape) == 3 and len(padded.shape) == 2:
        padded = padded[:, :, np.newaxis]

    return padded


def unpad_image(image, original_size=config.ORIG_SIZE):
    """
    Crops the center of the image to restore original dimensions.

    Args:
        image (np.ndarray): Padded image.
        original_size (int): Original height/width.

    Returns:
        np.ndarray: Unpadded image.
    """
    h, w = image.shape[:2]
    diff_h = h - original_size
    diff_w = w - original_size

    pad_top = diff_h // 2
    pad_left = diff_w // 2

    return image[pad_top : pad_top + original_size, pad_left : pad_left + original_size]


# =============================================================================
# Metrics
# =============================================================================


def calc_iou_batch(preds, labels):
    """
    Calculates IoU for a batch of predictions and labels.
    Handles empty masks correctly (IoU=1 if both empty, 0 if one empty).

    Args:
        preds (np.ndarray or torch.Tensor): Binary predictions (Batch, H, W).
        labels (np.ndarray or torch.Tensor): Binary ground truth (Batch, H, W).

    Returns:
        np.ndarray: IoU scores for each item in batch.
    """
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    # Flatten spatial dims: (Batch, H*W)
    preds_flat = preds.reshape(preds.shape[0], -1) > 0.5
    labels_flat = labels.reshape(labels.shape[0], -1) > 0.5

    intersection = np.logical_and(preds_flat, labels_flat).sum(axis=1)
    union = np.logical_or(preds_flat, labels_flat).sum(axis=1)

    # Initialize IoU array
    ious = np.zeros(preds.shape[0], dtype=np.float32)

    # Case 1: Union > 0 -> IoU = Intersection / Union
    mask_union = union > 0
    ious[mask_union] = intersection[mask_union] / union[mask_union]

    # Case 2: Union == 0 (Both empty) -> IoU = 1.0
    mask_empty = union == 0
    ious[mask_empty] = 1.0

    return ious


def calc_map_score(preds, labels, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Calculates the Mean Average Precision at different IoU thresholds.

    Args:
        preds (np.ndarray or torch.Tensor): Binary predictions.
        labels (np.ndarray or torch.Tensor): Binary ground truth.
        thresholds (np.ndarray): List of IoU thresholds.

    Returns:
        float: The mean average precision score.
    """
    ious = calc_iou_batch(preds, labels)

    # Calculate precision for each threshold
    # For a single image, precision is 1 if IoU > t, else 0
    # Average precision for the batch at threshold t is mean(IoU > t)

    precisions = []
    for t in thresholds:
        # Use a small epsilon for float comparison safety, though strictly > is defined
        matches = ious > t
        precisions.append(np.mean(matches))

    return np.mean(precisions)


# =============================================================================
# Data Loading & Caching
# =============================================================================


def preprocess_image(image_path):
    """
    Reads an image, collapses channels (if RGB), and pads it.
    """
    full_path = os.path.join(config.INPUT_DIR, image_path)
    # Read as grayscale directly or read unchanged and sum?
    # Task description says "summed RGB weights" in idea.
    # Let's read UNCHANGED to be safe, then process.
    img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Image not found: {full_path}")

    # Handle channels: if 3 channels, sum them to make 1 channel grayscale-like structure
    # If already 1 channel, keep it.
    if len(img.shape) == 3:
        # Summing channels as per idea description to retain info
        img = np.sum(img, axis=2, dtype=np.float32)
        # Normalize back to 0-255 range for storage efficiency or keep as float?
        # Standard approach: keep as float32 for training, but here we return raw values.
        # To fit in uint8 for cache we might lose info if sum > 255.
        # Let's normalize to 0-255 based on max value or just clip?
        # Actually, standard conversion is better. Let's stick to standard grayscale
        # unless specific "sum" logic requires float.
        # "Modify first conv layer... to accept 1-channel... by summing original RGB weights"
        # This usually refers to model weights. For input, we usually just convert to grayscale.
        # Let's use standard grayscale conversion to be safe and consistent.
        img = cv2.cvtColor(cv2.imread(full_path), cv2.COLOR_BGR2GRAY)

    # Pad
    img = pad_image(img, config.IMG_SIZE)
    return img


def preprocess_mask(mask_path):
    """
    Reads a mask and pads it.
    """
    full_path = os.path.join(config.INPUT_DIR, mask_path)
    mask = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {full_path}")

    # Pad
    mask = pad_image(mask, config.IMG_SIZE)
    # Ensure binary
    mask = (mask > 127).astype(np.float32)
    return mask


def load_dataset_data(
    metadata_path,
    cache_images_path,
    cache_masks_path=None,
    cache_depths_path=None,
    cache_ids_path=None,
    load_cached_data=True,
):
    """
    Loads dataset images, masks (optional), depths, and IDs.
    Uses caching to speed up subsequent loads.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_images_path (str): Path to save/load cached images .npy.
        cache_masks_path (str, optional): Path to save/load cached masks .npy.
        cache_depths_path (str, optional): Path to save/load cached depths .npy.
        cache_ids_path (str, optional): Path to save/load cached IDs .npy.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing 'images', 'masks', 'depths', 'ids'.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_images_path), exist_ok=True)

    has_masks = cache_masks_path is not None

    # Check if all required cache files exist
    cache_exists = (
        os.path.exists(cache_images_path)
        and (not has_masks or os.path.exists(cache_masks_path))
        and (cache_depths_path is None or os.path.exists(cache_depths_path))
        and (cache_ids_path is None or os.path.exists(cache_ids_path))
    )

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {os.path.dirname(cache_images_path)}...")
        data = {}
        data["images"] = np.load(cache_images_path)
        if has_masks:
            data["masks"] = np.load(cache_masks_path)
        if cache_depths_path:
            data["depths"] = np.load(cache_depths_path)
        if cache_ids_path:
            data["ids"] = np.load(cache_ids_path, allow_pickle=True)
        return data

    # Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    images = []
    masks = []
    depths = []
    ids = []

    for _, row in df.iterrows():
        # Load Image
        img = preprocess_image(row["image_path"])
        images.append(img)

        # Load Mask if applicable
        if has_masks:
            mask = preprocess_mask(row["mask_path"])
            masks.append(mask)

        # Load Depth
        if "z" in row:
            depths.append(row["z"])

        # Load ID
        ids.append(row["id"])

    # Convert to numpy arrays
    # Images: (N, H, W, 1) or (N, H, W). Let's standardize to (N, H, W, 1)
    images_arr = np.array(images, dtype=np.uint8)
    if len(images_arr.shape) == 3:
        images_arr = np.expand_dims(images_arr, axis=-1)

    # Save Images
    np.save(cache_images_path, images_arr)

    result = {"images": images_arr}

    if has_masks:
        masks_arr = np.array(masks, dtype=np.float32)
        if len(masks_arr.shape) == 3:
            masks_arr = np.expand_dims(masks_arr, axis=-1)
        np.save(cache_masks_path, masks_arr)
        result["masks"] = masks_arr

    if cache_depths_path and depths:
        depths_arr = np.array(depths, dtype=np.float32)
        np.save(cache_depths_path, depths_arr)
        result["depths"] = depths_arr

    if cache_ids_path and ids:
        ids_arr = np.array(ids)
        np.save(cache_ids_path, ids_arr)
        result["ids"] = ids_arr

    return result
