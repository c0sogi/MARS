import os
import struct
import numpy as np
import pandas as pd
import cv2
from library.config import Config


def parse_dicom_tags(file_path):
    """
    Scans a DICOM file binary for specific tags (Rows, Cols, PhotometricInterpretation)
    without parsing the entire file structure.
    """
    tags = {"rows": None, "cols": None, "photometric_interpretation": None}
    # Simplified to only look for Photometric Interpretation for inversion
    # Dimensions are now handled robustly via full read
    SIG_PHOTO = b"\x28\x00\x04\x00"

    try:
        with open(file_path, "rb") as f:
            header = f.read(4096)

            # Simple heuristic for Photometric Interpretation
            idx = header.find(SIG_PHOTO)
            if idx != -1:
                # This is a very rough heuristic, but sufficient for finding "MONOCHROME1"
                # if it appears shortly after the tag
                window = header[idx : idx + 50]
                if b"MONOCHROME1" in window:
                    tags["photometric_interpretation"] = "MONOCHROME1"
    except Exception:
        pass

    return tags


def fix_photometric_interpretation(image_array, photometric_interpretation):
    """
    Inverts the image if Photometric Interpretation is MONOCHROME1.
    MONOCHROME1: 0 = White, 1 = Black
    MONOCHROME2: 0 = Black, 1 = White (Standard)
    """
    if photometric_interpretation == "MONOCHROME1":
        # Invert based on data type range
        if image_array.dtype == np.uint8:
            return 255 - image_array
        elif image_array.dtype == np.uint16:
            return 65535 - image_array
        else:
            # Float or other, assume normalized or handle max
            return np.max(image_array) - image_array
    return image_array


def read_dicom_binary(file_path, fix_monochrome=True):
    """
    Reads a DICOM file using OpenCV for pixel data and binary scanning for metadata.
    Applies photometric correction if requested.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Read Pixel Data using OpenCV (Robust to compression)
    # flag -1 (IMREAD_UNCHANGED) preserves bit-depth (e.g. 16-bit)
    img = cv2.imread(file_path, -1)

    if img is None:
        return None

    # 2. Parse Metadata for Photometric Interpretation
    if fix_monochrome and Config.FIX_MONOCHROME1:
        tags = parse_dicom_tags(file_path)
        pi = tags.get("photometric_interpretation")
        if pi:
            img = fix_photometric_interpretation(img, pi)

    # Ensure image is 2D (H, W) or 3D (H, W, C)
    # If OpenCV loads as BGR (3 channels) but it's grayscale, take first channel
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Check if channels are identical (grayscale saved as RGB)
        if np.all(img[:, :, 0] == img[:, :, 1]):
            img = img[:, :, 0]

    return img


from concurrent.futures import ProcessPoolExecutor


def _get_dims_worker(args):
    """
    Worker function for parallel dimension extraction.
    """
    img_id, full_path = args
    try:
        # Robust read using OpenCV
        img = cv2.imread(full_path, -1)
        if img is not None:
            return img_id, img.shape[1], img.shape[0]
    except Exception:
        pass
    # Fallback only on total failure
    return img_id, 1024, 1024


def get_original_dimensions(df, load_cached_data=True):
    """
    Retrieves original dimensions (width, height) for images in the dataframe.
    Uses caching to avoid re-parsing headers.
    Cite Lesson 22: Coupling Geometric Metadata with Image Loading.
    We use robust full-image reading instead of heuristic parsing.
    """
    # Use a new cache file name to avoid loading potentially corrupt data from previous runs
    cache_file = os.path.join(Config.CACHE_DIR, "original_dimensions_safe.parquet")

    # 1. Load Cache
    cached_df = pd.DataFrame(columns=["image_id", "orig_width", "orig_height"])
    if load_cached_data and os.path.exists(cache_file):
        try:
            cached_df = pd.read_parquet(cache_file)
        except Exception:
            pass

    # 2. Identify Missing IDs
    df["image_id"] = df["image_id"].astype(str)
    cached_df["image_id"] = cached_df["image_id"].astype(str)

    unique_ids = df["image_id"].unique()
    known_ids = set(cached_df["image_id"].values)
    missing_ids = [uid for uid in unique_ids if uid not in known_ids]

    if len(missing_ids) == 0:
        return dict(
            zip(
                cached_df["image_id"],
                zip(cached_df["orig_width"], cached_df["orig_height"]),
            )
        )

    # 3. Process Missing (Parallelized)
    path_lookup = (
        df.drop_duplicates("image_id").set_index("image_id")["file_path"].to_dict()
    )

    tasks = []
    for img_id in missing_ids:
        rel_path = path_lookup.get(img_id)
        if rel_path:
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            tasks.append((img_id, full_path))

    new_rows = []
    if tasks:
        # Use parallel processing to speed up the robust read
        max_workers = min(12, os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_get_dims_worker, tasks))

        for r in results:
            new_rows.append({"image_id": r[0], "orig_width": r[1], "orig_height": r[2]})

    # 4. Update Cache
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([cached_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["image_id"], keep="last")

        if load_cached_data:
            combined_df.to_parquet(cache_file)
        cached_df = combined_df

    return dict(
        zip(
            cached_df["image_id"],
            zip(cached_df["orig_width"], cached_df["orig_height"]),
        )
    )


def rescale_boxes(boxes, current_shape, original_shape):
    """
    Rescales bounding boxes from current_shape to original_shape.
    boxes: Array-like of shape (N, 4) -> [xmin, ymin, xmax, ymax]
    current_shape: (height, width)
    original_shape: (height, width)
    """
    if len(boxes) == 0:
        return boxes

    boxes = np.array(boxes, dtype=np.float32)
    curr_h, curr_w = current_shape
    orig_h, orig_w = original_shape

    scale_x = orig_w / curr_w
    scale_y = orig_h / curr_h

    boxes[:, 0] *= scale_x  # xmin
    boxes[:, 1] *= scale_y  # ymin
    boxes[:, 2] *= scale_x  # xmax
    boxes[:, 3] *= scale_y  # ymax

    return boxes
