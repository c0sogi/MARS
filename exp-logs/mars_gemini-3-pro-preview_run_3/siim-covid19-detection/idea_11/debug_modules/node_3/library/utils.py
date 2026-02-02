import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import ast
from library.config import Config
from library.backbone import NestedTensor


def load_dicom(path):
    """
    Reads a DICOM file and returns a numpy array.
    Handles MONOCHROME1 inversion and normalization to 0-255.
    """
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)

        # Handle Photometric Interpretation
        if (
            hasattr(dcm, "PhotometricInterpretation")
            and dcm.PhotometricInterpretation == "MONOCHROME1"
        ):
            img = np.max(img) - img

        # Normalize to 0-255
        if np.max(img) != 0:
            img = img / np.max(img)
        img = (img * 255).astype(np.uint8)

        # Ensure 3 channels if needed (though usually grayscale is fine,
        # backbones often expect 3)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        return img
    except Exception as e:
        print(f"Error loading DICOM {path}: {e}")
        # Return a black image of default size in case of error
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)


def letterbox_resize(img, target_size=Config.IMG_SIZE):
    """
    Resizes image to target_size while maintaining aspect ratio.
    Pads with zeros.
    Returns:
        img: Resized and padded image
        ratio: Scale factor
        pad: (dw, dh) padding values
    """
    shape = img.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(target_size / shape[0], target_size / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = target_size - new_unpad[0], target_size - new_unpad[1]  # wh padding

    # Divide padding by 2 for centering
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )  # add border

    return img, r, (left, top)


def box_cxcywh_to_xyxy(x):
    """
    Converts bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2).
    x can be a torch tensor or numpy array.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """
    Converts bounding boxes from (x1, y1, x2, y2) to (cx, cy, w, h).
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def scale_coords(coords, ratio, pad, to_original=True):
    """
    Transforms coordinates between resized/padded image and original image.
    coords: (N, 4) tensor or array in format (x1, y1, x2, y2)
    ratio: scale factor used in resizing
    pad: (pad_w, pad_h) used in resizing
    to_original: If True, converts from resized->original. If False, original->resized.
    """
    pad_w, pad_h = pad

    if to_original:
        # Remove padding
        coords[:, [0, 2]] -= pad_w
        coords[:, [1, 3]] -= pad_h
        # Undo scaling
        coords[:, :4] /= ratio
    else:
        # Apply scaling
        coords[:, :4] *= ratio
        # Add padding
        coords[:, [0, 2]] += pad_w
        coords[:, [1, 3]] += pad_h

    return coords


def box_iou(boxes1, boxes2):
    """
    Computes IoU between two sets of boxes.
    boxes1: (N, 4)
    boxes2: (M, 4)
    Returns: (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def preprocess_metadata(csv_path, load_cached_data=True, split_name="train"):
    """
    Loads metadata CSV, processes bounding boxes, and handles caching.

    Args:
        csv_path: Path to the source CSV file.
        load_cached_data: Whether to try loading from cache.
        split_name: 'train', 'val', or 'test' to identify cache file.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"cached_{split_name}_df.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Ensure boxes are lists (parquet might save as array or list)
            if "boxes" in df.columns:
                # Check if the first non-null element is a string, if so, it needs parsing
                # Parquet usually preserves types, but let's be safe
                sample = (
                    df["boxes"].dropna().iloc[0]
                    if not df["boxes"].dropna().empty
                    else []
                )
                if isinstance(sample, str):
                    df["boxes"] = df["boxes"].apply(
                        lambda x: ast.literal_eval(x) if pd.notna(x) else []
                    )
                elif isinstance(sample, np.ndarray):
                    df["boxes"] = df["boxes"].apply(
                        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                    )

            print(f"Loaded {split_name} metadata from cache: {cache_path}")
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing.")

    # 2. Process from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source metadata not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse bounding boxes if they exist (train/val)
    if "boxes" in df.columns:
        # The 'boxes' column in the CSV is a string representation of a list of dicts
        # e.g., "[{'x': 10, 'y': 10, 'width': 50, 'height': 50}]" or NaN

        def parse_boxes(box_str):
            if pd.isna(box_str):
                return []
            try:
                # ast.literal_eval is safer than eval
                box_dicts = ast.literal_eval(box_str)
                # Convert to [x, y, w, h] list format
                boxes = []
                for b in box_dicts:
                    boxes.append([b["x"], b["y"], b["width"], b["height"]])
                return boxes
            except:
                return []

        df["boxes"] = df["boxes"].apply(parse_boxes)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    # Parquet doesn't support lists well in older pandas versions without pyarrow,
    # but recent versions handle it. If issues arise, we can pickle, but prompt said parquet/npy.
    # We will convert lists to strings for parquet storage if needed, but let's try direct save.
    # To be safe and compliant with "no pickle" preference for complex objects in some envs,
    # we can save as parquet.
    try:
        df.to_parquet(cache_path, index=False)
        print(f"Saved {split_name} metadata to cache: {cache_path}")
    except Exception as e:
        print(
            f"Warning: Could not save cache to parquet ({e}). Continuing without caching."
        )

    return df


def format_prediction_string(labels, boxes, scores):
    """
    Formats predictions for a single image into the submission string format.
    labels: list of class indices (0 for opacity)
    boxes: list of [x1, y1, x2, y2]
    scores: list of confidence scores
    """
    pred_strings = []
    for i, label in enumerate(labels):
        # We only have one class 'opacity' for image-level object detection
        # But for study level we have others. This function is primarily for image level.
        if label == -1:  # Convention for 'none'
            pred_strings.append("none 1 0 0 1 1")
        else:
            # Assuming label 0 is opacity
            class_name = "opacity"
            score = scores[i]
            x1, y1, x2, y2 = boxes[i]
            # Ensure coordinates are valid
            x1 = max(0, x1)
            y1 = max(0, y1)
            pred_strings.append(
                f"{class_name} {score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"
            )

    if not pred_strings:
        return "none 1 0 0 1 1"

    return " ".join(pred_strings)


def format_study_prediction(label_idx, score):
    """
    Formats study-level prediction.
    label_idx: 0-3 index corresponding to STUDY_LABELS
    score: confidence score
    """
    class_name = Config.STUDY_LABELS[label_idx]
    # Study prediction format: class_id confidence 0 0 1 1
    # Note: The prompt says "class ID from the above list". The list contains strings.
    # The example shows: "negative 1 0 0 1 1"

    # Map full names to submission format names if necessary.
    # Based on sample submission: "negative", "typical", "indeterminate", "atypical"
    # Config.STUDY_LABELS = ["Negative for Pneumonia", "Typical Appearance", ...]

    name_map = {
        "Negative for Pneumonia": "negative",
        "Typical Appearance": "typical",
        "Indeterminate Appearance": "indeterminate",
        "Atypical Appearance": "atypical",
    }

    short_name = name_map.get(class_name, "negative")
    return f"{short_name} {score:.4f} 0 0 1 1"


def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    Handles variable number of boxes per image.
    """
    batch = list(zip(*batch))
    images = batch[0]
    targets = batch[1]

    # Stack images to (B, C, H, W)
    images = torch.stack(images, dim=0)

    # Create mask (B, H, W) - False indicates valid pixel
    mask = torch.zeros(
        (images.shape[0], images.shape[2], images.shape[3]), dtype=torch.bool
    )

    return NestedTensor(images, mask), targets
