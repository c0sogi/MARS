import os
import struct
import numpy as np
import pandas as pd
import cv2
import torch
from library.config import Config


def read_dicom_binary(file_path):
    """
    Reads a DICOM file from a byte stream, parsing pixel data and metadata
    without using pydicom. Falls back to OpenCV if possible.
    Returns a numpy array of the image.
    """
    # 1. Try OpenCV
    # OpenCV can often handle DICOM if compiled with support (e.g. via Jasper/GDCM)
    try:
        img = cv2.imread(file_path, -1)
        if img is not None:
            # Ensure it's float32 for downstream processing
            return img.astype(np.float32)
    except Exception:
        pass

    # 2. Manual Binary Parsing (Fallback)
    # Scans for specific tags (Rows, Cols, PixelData) in the binary stream.
    # Assumes Little Endian Transfer Syntax (standard for this dataset).
    try:
        with open(file_path, "rb") as f:
            data = f.read()
    except IOError:
        # Return black image if file read fails
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    def get_tag_value(tag_bytes, data):
        """Finds a tag and returns its value (assuming US/SS/UL)."""
        offset = 0
        while True:
            idx = data.find(tag_bytes, offset)
            if idx == -1:
                return None

            # Heuristic: Check if what follows looks like a valid VR or Length
            # Look ahead 4 bytes (Tag) + 2 bytes (VR?)
            vr_offset = idx + 4
            if vr_offset + 6 > len(data):
                return None

            vr = data[vr_offset : vr_offset + 2]

            # Case 1: Explicit VR (e.g., b'US')
            if vr == b"US":
                # Tag(4) + VR(2) + Length(2) + Value(2)
                val_offset = vr_offset + 4
                return struct.unpack("<H", data[val_offset : val_offset + 2])[0]

            # Case 2: Implicit VR
            # Tag(4) + Length(4) + Value(...)
            # We check if the next 4 bytes look like a reasonable length (e.g., 2 or 4)
            try:
                length = struct.unpack("<I", data[vr_offset : vr_offset + 4])[0]
                if length == 2:
                    return struct.unpack("<H", data[vr_offset + 4 : vr_offset + 6])[0]
            except:
                pass

            offset = idx + 1

    # Search for Rows (0028, 0010) and Columns (0028, 0011)
    rows = get_tag_value(b"\x28\x00\x10\x00", data)
    cols = get_tag_value(b"\x28\x00\x11\x00", data)

    # Search for Pixel Data (7FE0, 0010)
    pixel_tag = b"\xe0\x7f\x10\x00"
    idx = data.find(pixel_tag)

    if rows and cols and idx != -1:
        # Determine header skip based on VR
        vr = data[idx + 4 : idx + 6]
        header_skip = 0

        # Explicit VR with 4-byte length (OB, OW, OF, SQ, UT, UN)
        if vr in [b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"]:
            header_skip = 12  # Tag(4) + VR(2) + Reserved(2) + Length(4)
        # Explicit VR with 2-byte length
        elif vr.isalpha() and vr.isupper():
            header_skip = 8  # Tag(4) + VR(2) + Length(2)
        else:
            header_skip = 8  # Implicit VR: Tag(4) + Length(4)

        pixel_start = idx + header_skip
        expected_pixels = rows * cols

        # Try reading as uint16 (common for medical images)
        if pixel_start + expected_pixels * 2 <= len(data):
            pixel_bytes = data[pixel_start : pixel_start + expected_pixels * 2]
            img_array = np.frombuffer(pixel_bytes, dtype=np.uint16).reshape(
                (rows, cols)
            )
            return img_array.astype(np.float32)

        # Try reading as uint8
        elif pixel_start + expected_pixels <= len(data):
            pixel_bytes = data[pixel_start : pixel_start + expected_pixels]
            img_array = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape((rows, cols))
            return img_array.astype(np.float32)

    # Fallback if parsing fails
    return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def get_original_dimensions(df, load_cached_data=True):
    """
    Retrieves original image dimensions (width, height) for all images in the dataframe.
    Uses caching to avoid re-reading files.

    Args:
        df: DataFrame containing 'image_id' and 'file_path'.
        load_cached_data: Boolean, whether to attempt loading from cache.

    Returns:
        Dictionary mapping image_id to (width, height).
    """
    cache_path = os.path.join(Config.CACHE_DIR, "original_dimensions.parquet")

    # 1. Attempt Load from Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            # Check if cache covers the requested dataframe
            cached_ids = set(cache_df["image_id"])
            requested_ids = set(df["image_id"])

            if requested_ids.issubset(cached_ids):
                mapping = {}
                for _, row in cache_df.iterrows():
                    mapping[row["image_id"]] = (row["width"], row["height"])
                return mapping
        except Exception:
            # If cache load fails, proceed to compute
            pass

    # 2. Compute Dimensions
    results = []
    # Process unique images only
    unique_df = df.drop_duplicates(subset=["image_id"])

    for _, row in unique_df.iterrows():
        img_id = row["image_id"]
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if os.path.exists(full_path):
            img = read_dicom_binary(full_path)
            h, w = img.shape[:2]
        else:
            # Should not happen given EDA, but safe fallback
            h, w = Config.IMG_SIZE, Config.IMG_SIZE

        results.append({"image_id": img_id, "width": w, "height": h})

    result_df = pd.DataFrame(results)

    # 3. Update Cache
    # If cache exists, merge to preserve data for other splits
    if os.path.exists(cache_path):
        try:
            old_cache = pd.read_parquet(cache_path)
            result_df = pd.concat([old_cache, result_df]).drop_duplicates(
                subset=["image_id"]
            )
        except Exception:
            pass

    result_df.to_parquet(cache_path, index=False)

    # 4. Return Dictionary
    mapping = {}
    for _, row in result_df.iterrows():
        mapping[row["image_id"]] = (row["width"], row["height"])
    return mapping


def save_checkpoint(model, optimizer, scheduler, epoch, score, path):
    """
    Saves the model checkpoint.
    """
    # Ensure score is a standard float to avoid pickling numpy scalars
    if hasattr(score, "item"):
        score = score.item()
    score = float(score)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "score": score,
        },
        path,
    )


def load_checkpoint(model, path, optimizer=None, scheduler=None):
    """
    Loads the model checkpoint.
    Returns: (epoch, score)
    """
    if not os.path.exists(path):
        return 0, 0.0

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("score", 0.0)
