import os
import numpy as np
import pandas as pd
import cv2
from library.config import Config, seed_everything


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    Wrapper around the config's seed_everything function.
    """
    seed_everything(seed)


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    The competition format requires a space-delimited list of pairs (start, length).
    Pixels are one-indexed and numbered from top to bottom, then left to right (Fortran order).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 1 - salt, 0 - background.

    Returns:
        str: RLE string.
    """
    # Flatten column-major (Fortran style) as per competition spec
    pixels = mask.flatten(order="F")

    # We need to find the start and end of runs of 1s
    # Prepend and append 0 to detect starts and ends at boundaries
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is start of first run, runs[1] is end of first run (exclusive)
    # The length is runs[1] - runs[0]
    # We need to transform runs[1::2] (the ends) into lengths
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask.
    """
    if not isinstance(mask_rle, str) or pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # Convert 1-indexed to 0-indexed
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-major to match the encoding direction
    return img.reshape(shape, order="F")


def pad_image(image, target_size=128):
    """
    Pads an image to the target size using reflection padding.
    Handles both (H, W) and (H, W, 1) inputs.

    Args:
        image (np.ndarray): Input image.
        target_size (int): Target height/width (assumes square).

    Returns:
        np.ndarray: Padded image.
    """
    h, w = image.shape[:2]
    delta_h = target_size - h
    delta_w = target_size - w

    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # Handle channel dimension for cv2
    squeeze = False
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
        squeeze = True

    # cv2.BORDER_REFLECT_101 is standard for this task to avoid artifacts
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)

    if squeeze:
        padded = np.expand_dims(padded, axis=-1)
    elif len(image.shape) == 3:
        # If it was 3 channels (e.g. RGB), cv2 preserves it, but we handle the 1-channel case explicitly above
        pass

    return padded


def crop_image(image, original_size=101):
    """
    Center crops an image back to the original size.

    Args:
        image (np.ndarray): Input image (H, W) or (H, W, C).
        original_size (int): Target height/width (assumes square).

    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape[:2]
    delta_h = h - original_size
    delta_w = w - original_size

    top = delta_h // 2
    left = delta_w // 2

    if len(image.shape) == 3:
        return image[top : top + original_size, left : left + original_size, :]
    else:
        return image[top : top + original_size, left : left + original_size]


def load_data_and_cache(df, dataset_type="train", load_cached_data=True):
    """
    Loads images and masks (if available) from the dataframe.
    Implements caching mechanism to save processed numpy arrays to disk.

    Args:
        df (pd.DataFrame): Dataframe containing 'image_path' and optionally 'mask_path'.
        dataset_type (str): Name for the cache file prefix (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing 'images', 'ids', 'depths', and optionally 'masks'.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    images_path = os.path.join(cache_dir, f"{dataset_type}_images.npy")
    masks_path = os.path.join(cache_dir, f"{dataset_type}_masks.npy")
    ids_path = os.path.join(cache_dir, f"{dataset_type}_ids.npy")
    depths_path = os.path.join(cache_dir, f"{dataset_type}_depths.npy")

    has_masks = "mask_path" in df.columns and df["mask_path"].notna().any()

    # 1. Try to load from cache
    if load_cached_data:
        try:
            # Check if required files exist
            files_exist = (
                os.path.exists(images_path)
                and os.path.exists(ids_path)
                and os.path.exists(depths_path)
            )
            if has_masks:
                files_exist = files_exist and os.path.exists(masks_path)

            if files_exist:
                # Load data
                images = np.load(images_path)
                ids = np.load(
                    ids_path
                )  # Assumes saved as fixed-type string or compatible
                depths = np.load(depths_path)

                result = {"images": images, "ids": ids, "depths": depths}

                if has_masks:
                    masks = np.load(masks_path)
                    result["masks"] = masks

                # Simple consistency check
                if len(images) == len(df):
                    return result
                # If lengths differ (e.g. debug subset vs full cache), recompute
        except Exception:
            # If loading fails for any reason, proceed to compute
            pass

    # 2. Compute from scratch
    img_list = []
    mask_list = []
    id_list = []
    depth_list = []

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback for safety, though metadata validation should prevent this
            img = np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)

        img_list.append(img)
        id_list.append(row["id"])
        depth_list.append(row["z"])

        # Load Mask if exists
        if has_masks:
            msk_path_rel = row["mask_path"]
            if pd.isna(msk_path_rel):
                msk = np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)
            else:
                msk_path = os.path.join(Config.INPUT_DIR, msk_path_rel)
                msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
                if msk is None:
                    msk = np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)
                else:
                    # Ensure binary
                    msk = (msk > 127).astype(np.uint8)
            mask_list.append(msk)

    # Convert to numpy arrays
    # Images: (N, 101, 101, 1)
    images = np.array(img_list, dtype=np.uint8)
    images = np.expand_dims(images, axis=-1)

    # IDs: Use unicode fixed length to avoid pickle requirement
    ids = np.array(id_list, dtype="U10")

    # Depths: Float32
    depths = np.array(depth_list, dtype=np.float32)

    # Save to cache
    np.save(images_path, images)
    np.save(ids_path, ids)
    np.save(depths_path, depths)

    result = {"images": images, "ids": ids, "depths": depths}

    if has_masks:
        masks = np.array(mask_list, dtype=np.uint8)
        masks = np.expand_dims(masks, axis=-1)
        np.save(masks_path, masks)
        result["masks"] = masks

    return result
