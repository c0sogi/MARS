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
    Assumes Little Endian Explicit VR for simplicity, which covers most datasets.
    """
    tags = {"rows": None, "cols": None, "photometric_interpretation": None}

    # Byte signatures for tags (Group, Element) in Little Endian
    # (0028, 0010) -> Rows
    SIG_ROWS = b"\x28\x00\x10\x00"
    # (0028, 0011) -> Columns
    SIG_COLS = b"\x28\x00\x11\x00"
    # (0028, 0004) -> Photometric Interpretation
    SIG_PHOTO = b"\x28\x00\x04\x00"

    try:
        with open(file_path, "rb") as f:
            # Read the first 4KB, usually enough for headers
            header = f.read(4096)

            # Helper to extract value based on signature
            def find_tag(signature, vr_type):
                idx = header.find(signature)
                if idx == -1:
                    return None

                # Skip Tag (4 bytes)
                current = idx + 4

                # Read VR (2 bytes)
                vr = header[current : current + 2]
                current += 2

                # Explicit VR handling
                length = 0
                if vr in [b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"]:
                    current += 2  # Skip reserved
                    length = struct.unpack("<I", header[current : current + 4])[0]
                    current += 4
                else:
                    length = struct.unpack("<H", header[current : current + 2])[0]
                    current += 2

                value_bytes = header[current : current + length]

                if vr_type == "US":  # Unsigned Short
                    return struct.unpack("<H", value_bytes)[0]
                elif vr_type == "CS":  # Code String
                    return value_bytes.decode("utf-8", errors="ignore").strip()
                return None

            tags["rows"] = find_tag(SIG_ROWS, "US")
            tags["cols"] = find_tag(SIG_COLS, "US")
            tags["photometric_interpretation"] = find_tag(SIG_PHOTO, "CS")

    except Exception as e:
        # Fallback or silent fail - will be handled by consumer
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
        raise ValueError(f"Failed to read image data from {file_path}")

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


def get_original_dimensions(df, load_cached_data=True):
    """
    Retrieves original dimensions (width, height) for images in the dataframe.
    Uses caching to avoid re-parsing headers.
    """
    cache_file = os.path.join(Config.CACHE_DIR, "original_dimensions.parquet")

    # 1. Load Cache
    cached_df = pd.DataFrame(columns=["image_id", "orig_width", "orig_height"])
    if load_cached_data and os.path.exists(cache_file):
        try:
            cached_df = pd.read_parquet(cache_file)
        except Exception:
            pass  # Corrupt cache, start over

    # 2. Identify Missing IDs
    # Ensure image_id is string
    df["image_id"] = df["image_id"].astype(str)
    cached_df["image_id"] = cached_df["image_id"].astype(str)

    # Filter for unique image_ids in input df
    unique_ids = df["image_id"].unique()

    # Check what is already in cache
    known_ids = set(cached_df["image_id"].values)
    missing_ids = [uid for uid in unique_ids if uid not in known_ids]

    if len(missing_ids) == 0:
        # All present, return map
        return dict(
            zip(
                cached_df["image_id"],
                zip(cached_df["orig_width"], cached_df["orig_height"]),
            )
        )

    # 3. Process Missing
    new_rows = []

    # Create a lookup for file paths from the input df
    # We drop duplicates to have one path per image_id
    path_lookup = (
        df.drop_duplicates("image_id").set_index("image_id")["file_path"].to_dict()
    )

    for img_id in missing_ids:
        rel_path = path_lookup.get(img_id)
        if not rel_path:
            continue

        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Try fast tag parsing first
        tags = parse_dicom_tags(full_path)
        h, w = tags.get("rows"), tags.get("cols")

        # Fallback to loading image if tags failed
        if h is None or w is None:
            try:
                img = cv2.imread(full_path, -1)
                if img is not None:
                    h, w = img.shape[:2]
            except:
                pass

        # Default fallback if everything fails
        if h is None or w is None:
            h, w = 1024, 1024  # Placeholder

        new_rows.append({"image_id": img_id, "orig_width": w, "orig_height": h})

    # 4. Update Cache
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([cached_df, new_df], ignore_index=True)
        # Drop duplicates just in case
        combined_df = combined_df.drop_duplicates(subset=["image_id"], keep="last")

        # Save
        if load_cached_data:
            combined_df.to_parquet(cache_file)

        cached_df = combined_df

    # 5. Return Dictionary
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
