import os
import random
import numpy as np
import torch
import cv2
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, Numpy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE).
    The mask is flattened in column-major order (top-to-bottom, then left-to-right).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten column-major
    pixels = mask.T.flatten()
    # Pad with 0s at ends to detect changes at boundaries
    pixels = np.concatenate([[0], pixels, [0]])
    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(Config.ORIG_SIZE, Config.ORIG_SIZE)):
    """
    Decodes a Run-Length Encoded string to a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if not isinstance(mask_rle, str) or str(mask_rle) == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-major (W, H) then transpose to (H, W)
    return img.reshape((shape[1], shape[0])).T


def pad_image(image, target_size=Config.IMG_SIZE):
    """
    Pads an image using reflection padding to the target size.
    Assumes input is (H, W) or (H, W, C).
    """
    h, w = image.shape[:2]
    diff = target_size - h
    pad_top = diff // 2
    pad_bottom = diff - pad_top
    pad_left = diff // 2
    pad_right = diff - pad_left

    return cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )


def unpad_image(image, original_size=Config.ORIG_SIZE):
    """
    Crops the center of the image to return to original size.
    """
    h, w = image.shape[:2]
    diff = h - original_size
    pad_top = diff // 2
    return image[pad_top : pad_top + original_size, pad_top : pad_top + original_size]


def calculate_iou_batch(preds, gts):
    """
    Calculates IoU for a batch of predictions and ground truths.
    Handles the case where both are empty (IoU = 1.0).

    Args:
        preds (torch.Tensor): Binary predictions (B, H, W) or (B, 1, H, W).
        gts (torch.Tensor): Binary ground truth (B, H, W) or (B, 1, H, W).

    Returns:
        torch.Tensor: IoU scores for each item in the batch (B,).
    """
    # Flatten spatial dimensions
    p_flat = preds.view(preds.size(0), -1).float()
    g_flat = gts.view(gts.size(0), -1).float()

    intersection = (p_flat * g_flat).sum(1)
    union = p_flat.sum(1) + g_flat.sum(1) - intersection

    # Initialize IoU
    iou = torch.zeros_like(intersection)

    # Case: Union > 0
    non_empty = union > 0
    iou[non_empty] = intersection[non_empty] / union[non_empty]

    # Case: Union == 0 (Both empty) -> IoU = 1.0
    iou[~non_empty] = 1.0

    return iou


def compute_map_score(preds, gts, thresholds=Config.IOU_THRESHOLDS):
    """
    Computes the Mean Average Precision (mAP) over specified IoU thresholds.

    Args:
        preds (torch.Tensor): Binary predictions.
        gts (torch.Tensor): Binary ground truth.
        thresholds (list): List of IoU thresholds.

    Returns:
        float: The mean average precision score.
    """
    # Ensure inputs are tensors on the correct device
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(gts, torch.Tensor):
        gts = torch.tensor(gts)

    ious = calculate_iou_batch(preds, gts)  # Shape: (B,)

    # Calculate precision at each threshold
    # Precision is 1 if IoU > t, else 0 (per image)
    acc = []
    for t in thresholds:
        acc.append((ious > t).float().mean().item())

    return np.mean(acc)


def load_data(mode="train", load_cached_data=True):
    """
    Loads dataset images, masks, and depths. Uses caching to speed up subsequent runs.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, masks, depths, ids)
               images: (N, 128, 128) uint8
               masks: (N, 128, 128) float32 (0.0 or 1.0) or None for test
               depths: (N,) float32
               ids: (N,) str
    """
    # Define Cache Paths based on mode
    if mode == "train":
        p_img = Config.CACHE_TRAIN_IMAGES
        p_mask = Config.CACHE_TRAIN_MASKS
        p_depth = Config.CACHE_TRAIN_DEPTHS
        p_ids = os.path.join(Config.CACHE_DIR, "train_ids.npy")
        meta_path = Config.TRAIN_METADATA_PATH
    elif mode == "val":
        p_img = Config.CACHE_VAL_IMAGES
        p_mask = Config.CACHE_VAL_MASKS
        p_depth = Config.CACHE_VAL_DEPTHS
        p_ids = os.path.join(Config.CACHE_DIR, "val_ids.npy")
        meta_path = Config.VAL_METADATA_PATH
    elif mode == "test":
        p_img = Config.CACHE_TEST_IMAGES
        p_mask = None  # No masks for test
        p_depth = Config.CACHE_TEST_DEPTHS
        p_ids = Config.CACHE_TEST_IDS
        meta_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Attempt to load from cache
    if load_cached_data:
        try:
            images = np.load(p_img)
            depths = np.load(p_depth)
            ids = np.load(p_ids)
            masks = np.load(p_mask) if p_mask else None
            return images, masks, depths, ids
        except (FileNotFoundError, OSError):
            pass  # Fallback to processing

    # Process from scratch
    df = pd.read_csv(meta_path)

    # Pre-allocate arrays
    n_samples = len(df)
    images = np.zeros((n_samples, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)
    depths = df["z"].values.astype(np.float32)
    ids = df["id"].values

    if mode != "test":
        masks = np.zeros(
            (n_samples, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )
    else:
        masks = None

    for i, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # Fallback for missing file (should be caught by metadata check)
            img = np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)

        # Pad Image
        img_padded = pad_image(img)
        images[i] = img_padded

        # Load Mask (if applicable)
        if mode != "test":
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                mask = np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)

            # Pad Mask
            mask_padded = pad_image(mask)
            # Normalize to [0, 1] float
            masks[i] = (mask_padded > 127).astype(np.float32)

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(p_img, images)
    np.save(p_depth, depths)
    np.save(p_ids, ids)
    if masks is not None:
        np.save(p_mask, masks)

    return images, masks, depths, ids
