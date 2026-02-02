import os
import cv2
import numpy as np
import pydicom
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def load_dicom(path):
    """
    Reads a DICOM file and returns a numpy array (H, W, C) in RGB format (0-255).
    Handles path resolution, photometric interpretation, and normalization.
    """
    # Resolve full path if relative path is provided
    if not os.path.isabs(path) and not path.startswith(Config.INPUT_DIR):
        full_path = os.path.join(Config.INPUT_DIR, path)
    else:
        full_path = path

    try:
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array.astype(np.float32)

        # Handle MONOCHROME1 (bones are black, air is white) -> invert to standard MONOCHROME2
        if (
            hasattr(dcm, "PhotometricInterpretation")
            and dcm.PhotometricInterpretation == "MONOCHROME1"
        ):
            img = np.max(img) - img

        # Normalize to 0-255
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255.0
        else:
            img = np.zeros_like(img)

        img = img.astype(np.uint8)

        # Convert to RGB (3 channels) for backbone compatibility
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img
    except Exception as e:
        print(f"Error loading DICOM {full_path}: {e}")
        # Return a black image of default size if failure to prevent crash
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)


def letterbox_resize(img, target_size=Config.IMG_SIZE, boxes=None):
    """
    Resizes image to target_size while preserving aspect ratio and padding.
    If boxes are provided (xyxy format), they are also transformed.

    Args:
        img: Input image (H, W, C)
        target_size: Target dimension (int)
        boxes: Optional numpy array of bounding boxes [[x1, y1, x2, y2], ...]

    Returns:
        img: Resized and padded image.
        boxes: Transformed boxes (if input boxes provided).
        info: Dictionary containing 'scale', 'pad_w', 'pad_h' for restoring coordinates.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)

    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    dw = target_size - new_w
    dh = target_size - new_h

    # Center padding
    pad_l = dw // 2
    pad_r = dw - pad_l
    pad_t = dh // 2
    pad_b = dh - pad_t

    # Apply padding
    img_padded = cv2.copyMakeBorder(
        img_resized, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    info = {
        "scale": scale,
        "pad_w": pad_l,
        "pad_h": pad_t,
        "orig_h": h,
        "orig_w": w,
        "new_h": new_h,
        "new_w": new_w,
    }

    if boxes is not None and len(boxes) > 0:
        boxes = np.array(boxes, dtype=np.float32)
        # Scale
        boxes[:, [0, 2]] *= scale
        boxes[:, [1, 3]] *= scale
        # Shift
        boxes[:, [0, 2]] += pad_l
        boxes[:, [1, 3]] += pad_t
        return img_padded, boxes, info

    return img_padded, info


def scale_coords(coords, info):
    """
    Rescales boxes from letterbox image back to original image coordinates.

    Args:
        coords: Numpy array of boxes [[x1, y1, x2, y2], ...]
        info: Dictionary from letterbox_resize containing scale and pads.

    Returns:
        coords: Rescaled boxes in original image coordinates.
    """
    # Remove padding
    coords[:, [0, 2]] -= info["pad_w"]
    coords[:, [1, 3]] -= info["pad_h"]

    # Undo scaling
    coords[:, :4] /= info["scale"]

    # Clip to original dimensions
    coords[:, 0] = np.clip(coords[:, 0], 0, info["orig_w"])
    coords[:, 1] = np.clip(coords[:, 1], 0, info["orig_h"])
    coords[:, 2] = np.clip(coords[:, 2], 0, info["orig_w"])
    coords[:, 3] = np.clip(coords[:, 3], 0, info["orig_h"])

    return coords


def get_train_transforms():
    """
    Returns Albumentations transforms for training.
    Includes Large Scale Jittering (LSJ) simulation and flips.
    """
    return A.Compose(
        [
            # Large Scale Jittering: Resize to random scale, then crop/pad to target size
            A.RandomScale(
                scale_limit=(Config.AUG_MIN_SCALE - 1.0, Config.AUG_MAX_SCALE - 1.0),
                p=1.0,
            ),
            A.PadIfNeeded(
                min_height=Config.IMG_SIZE,
                min_width=Config.IMG_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            A.RandomCrop(height=Config.IMG_SIZE, width=Config.IMG_SIZE, p=1.0),
            # Horizontal Flip
            A.HorizontalFlip(p=Config.AUG_HFLIP_PROB),
            # Normalize and Convert to Tensor
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"], min_visibility=0.1
        ),
    )


def get_valid_transforms():
    """
    Returns Albumentations transforms for validation/inference.
    Implements deterministic letterbox resizing using Albumentations ops.
    """
    return A.Compose(
        [
            # Resize longest side to target size
            A.LongestMaxSize(max_size=Config.IMG_SIZE, interpolation=cv2.INTER_LINEAR),
            # Pad remaining dimensions to reach target square
            A.PadIfNeeded(
                min_height=Config.IMG_SIZE,
                min_width=Config.IMG_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            # Normalize and Convert to Tensor
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )


def collate_fn(batch):
    """
    Custom collate function for object detection batches.
    Handles variable number of boxes per image.

    Args:
        batch: List of tuples (image, target, image_id)
    """
    images = []
    targets = []
    image_ids = []

    for b in batch:
        images.append(b[0])
        targets.append(b[1])
        image_ids.append(b[2])

    images = torch.stack(images, dim=0)

    return images, targets, image_ids


def format_prediction_string(study_preds, image_preds):
    """
    Formats predictions into the submission string format.

    Args:
        study_preds: List of dictionaries for study predictions.
                     Each dict: {'id': study_id, 'class_id': int, 'conf': float}
        image_preds: List of dictionaries for image predictions.
                     Each dict: {'id': image_id, 'boxes': [[x1, y1, x2, y2], ...], 'scores': [s1, ...], 'study_neg': bool}

    Returns:
        List of strings ["id,PredictionString", ...]
    """
    lines = ["Id,PredictionString"]

    # Map class index to label string
    # 0: Negative, 1: Typical, 2: Indeterminate, 3: Atypical
    study_label_map = {0: "negative", 1: "typical", 2: "indeterminate", 3: "atypical"}

    # Process Study Level Predictions
    for pred in study_preds:
        s_id = pred["id"]
        if not s_id.endswith("_study"):
            s_id = f"{s_id}_study"

        label_str = study_label_map.get(pred["class_id"], "negative")
        conf = pred["conf"]
        # Format: label confidence 0 0 1 1
        pred_str = f"{label_str} {conf:.6f} 0 0 1 1"
        lines.append(f"{s_id},{pred_str}")

    # Process Image Level Predictions
    for pred in image_preds:
        i_id = pred["id"]
        if not i_id.endswith("_image"):
            i_id = f"{i_id}_image"

        # If study prediction was "Negative for Pneumonia" (class 0), force "none"
        if pred.get("study_neg", False):
            pred_str = "none 1 0 0 1 1"
        else:
            boxes = pred["boxes"]
            scores = pred["scores"]

            if len(boxes) == 0:
                pred_str = "none 1 0 0 1 1"
            else:
                parts = []
                for box, score in zip(boxes, scores):
                    # box is x1, y1, x2, y2
                    x1, y1, x2, y2 = box
                    parts.append(
                        f"opacity {score:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
                    )
                pred_str = " ".join(parts)

        lines.append(f"{i_id},{pred_str}")

    return lines
