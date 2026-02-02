import os
import cv2
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility using the Config class.

    Args:
        seed (int, optional): The seed to set. If None, uses Config.SEED.
    """
    if seed is None:
        seed = Config.SEED
    Config.set_seed(seed)


def rle_encode(mask):
    """
    Encodes a binary mask into RLE string format.
    Pixels are 1-indexed, top-to-bottom, then left-to-right.

    Args:
        mask (np.array): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited RLE string.
    """
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(rle_string, shape=(101, 101)):
    """
    Decodes an RLE string into a binary mask.

    Args:
        rle_string (str): Space-delimited RLE string.
        shape (tuple): Output shape (H, W).

    Returns:
        np.array: Binary mask of shape (H, W).
    """
    if pd.isna(rle_string) or rle_string == "":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def pad_image(image, target_size=128):
    """
    Pads an image or mask to the target size using reflection padding.

    Args:
        image (np.array): Image of shape (H, W) or (H, W, C).
        target_size (int): Target height/width.

    Returns:
        np.array: Padded image.
    """
    h, w = image.shape[:2]
    pad_h = target_size - h
    pad_w = target_size - w

    if pad_h < 0 or pad_w < 0:
        return image

    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return cv2.copyMakeBorder(
        image, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )


def unpad_image(image, original_size=101):
    """
    Crops the image or mask back to the original size (center crop).

    Args:
        image (np.array): Padded image.
        original_size (int): Original height/width.

    Returns:
        np.array: Cropped image.
    """
    h, w = image.shape[:2]
    pad_h = h - original_size
    pad_w = w - original_size

    if pad_h < 0 or pad_w < 0:
        return image

    pad_top = pad_h // 2
    pad_left = pad_w // 2

    return image[pad_top : pad_top + original_size, pad_left : pad_left + original_size]


def get_score(preds, targets, threshold_range=None):
    """
    Calculates the mean Average Precision (mAP) at IoU thresholds.

    Args:
        preds: Predicted masks (N, H, W) or (H, W), boolean or 0/1.
        targets: Ground truth masks (N, H, W) or (H, W), boolean or 0/1.
        threshold_range: List or array of thresholds. Default 0.5 to 0.95 step 0.05.

    Returns:
        float: The mean average precision.
    """
    if threshold_range is None:
        threshold_range = np.arange(0.5, 1.0, 0.05)

    # Ensure inputs are numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Binarize
    preds = (preds > 0.5).astype(np.uint8)
    targets = (targets > 0.5).astype(np.uint8)

    # Handle single image case
    if preds.ndim == 2:
        preds = preds[np.newaxis, ...]
        targets = targets[np.newaxis, ...]

    ious = []
    for p, t in zip(preds, targets):
        intersection = (p & t).sum()
        union = (p | t).sum()

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
        ious.append(iou)

    ious = np.array(ious)

    # Calculate mean precision over thresholds
    scores = []
    for t in threshold_range:
        matches = ious > t
        scores.append(np.mean(matches))

    return np.mean(scores)


def load_dataset_with_cache(df, cache_name, load_cached_data=True, subset_size=None):
    """
    Loads images and masks, applies padding, and caches the result.

    Args:
        df: DataFrame containing metadata.
        cache_name: Unique identifier for the cache file.
        load_cached_data: Boolean, whether to try loading from cache.
        subset_size: Integer, if set, limits the data size.

    Returns:
        dict: {'images': np.array, 'masks': np.array, 'depths': np.array, 'ids': np.array}
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Append subset size to cache name if applicable to avoid collisions
    if subset_size is not None:
        cache_name = f"{cache_name}_sub{subset_size}"
        if subset_size < len(df):
            df = df.iloc[:subset_size].copy()

    img_cache_path = os.path.join(cache_dir, f"{cache_name}_images.npy")
    mask_cache_path = os.path.join(cache_dir, f"{cache_name}_masks.npy")
    depth_cache_path = os.path.join(cache_dir, f"{cache_name}_depths.npy")
    ids_cache_path = os.path.join(cache_dir, f"{cache_name}_ids.npy")

    has_masks = "rle_mask" in df.columns

    # Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(img_cache_path)
            and os.path.exists(depth_cache_path)
            and os.path.exists(ids_cache_path)
        ):
            if not has_masks or os.path.exists(mask_cache_path):
                images = np.load(img_cache_path)
                depths = np.load(depth_cache_path)
                ids = np.load(ids_cache_path, allow_pickle=True)
                masks = np.load(mask_cache_path) if has_masks else None
                return {"images": images, "masks": masks, "depths": depths, "ids": ids}

    images = []
    masks = []
    depths = []
    ids = []

    for idx, row in df.iterrows():
        img_path = os.path.join(Config.INPUT_ROOT, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Pad and normalize
        img_padded = pad_image(img, Config.PAD_SIZE)
        img_padded = img_padded.astype(np.float32) / 255.0
        img_padded = np.expand_dims(img_padded, axis=0)  # (1, H, W)

        images.append(img_padded)
        depths.append(row["z"])
        ids.append(row["id"])

        if has_masks:
            mask = rle_decode(row["rle_mask"], (Config.ORIG_SIZE, Config.ORIG_SIZE))
            mask_padded = pad_image(mask, Config.PAD_SIZE)
            mask_padded = (mask_padded > 0).astype(np.float32)
            mask_padded = np.expand_dims(mask_padded, axis=0)  # (1, H, W)
            masks.append(mask_padded)

    images = np.array(images)
    depths = np.array(depths)
    ids = np.array(ids)

    np.save(img_cache_path, images)
    np.save(depth_cache_path, depths)
    np.save(ids_cache_path, ids)

    if has_masks:
        masks = np.array(masks)
        np.save(mask_cache_path, masks)
    else:
        masks = None

    return {"images": images, "masks": masks, "depths": depths, "ids": ids}
