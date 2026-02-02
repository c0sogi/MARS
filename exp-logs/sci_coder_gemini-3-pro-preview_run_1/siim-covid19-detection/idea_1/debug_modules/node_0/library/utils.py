import os
import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import ast
import torch
from library.config import Config

# Constants for Class Mapping
STUDY_CLASSES = ["negative", "typical", "indeterminate", "atypical"]


def read_dicom(file_path, img_size=Config.IMG_SIZE, fix_monochrome=True):
    """
    Reads a DICOM file, applies VOI LUT, fixes monochrome interpretation,
    resizes, and converts to RGB.

    Args:
        file_path (str): Path to the DICOM file.
        img_size (int): Target spatial dimension (square).
        fix_monochrome (bool): Whether to invert MONOCHROME1 images.

    Returns:
        np.ndarray: Preprocessed image of shape (img_size, img_size, 3), uint8.
    """
    try:
        dcm = pydicom.dcmread(file_path)

        # Apply VOI LUT (Value of Interest Look-Up Table) if available
        # This handles window center/width transformations
        pixel_array = apply_voi_lut(dcm.pixel_array, dcm)

        # Fix Photometric Interpretation
        # MONOCHROME1: 0 = White, 1 = Black. We want 0 = Black (Air), 1 = White (Bone/Tissue)
        if fix_monochrome and dcm.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = np.amax(pixel_array) - pixel_array

        # Normalize to 0-255
        pixel_array = pixel_array - np.min(pixel_array)
        max_val = np.max(pixel_array)
        if max_val > 0:
            pixel_array = pixel_array / max_val
        pixel_array = (pixel_array * 255).astype(np.uint8)

        # Resize
        if img_size is not None:
            pixel_array = cv2.resize(pixel_array, (img_size, img_size))

        # Convert to 3 channels (RGB) for ResNet backbone
        # Stack along the last axis
        img_rgb = np.stack([pixel_array, pixel_array, pixel_array], axis=-1)

        return img_rgb

    except Exception as e:
        # Fallback for corrupt files or errors, return black image
        print(f"Error reading DICOM {file_path}: {e}")
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)


def create_mask(boxes_str, img_size=Config.IMG_SIZE, orig_w=1, orig_h=1):
    """
    Creates a binary mask from bounding box annotations.

    Args:
        boxes_str (str): String representation of list of dicts (e.g., "[{'x': 10...}]").
        img_size (int): Target mask dimension.
        orig_w (int): Original image width.
        orig_h (int): Original image height.

    Returns:
        np.ndarray: Binary mask of shape (img_size, img_size), float32.
    """
    mask = np.zeros((img_size, img_size), dtype=np.float32)

    # Handle NaN or empty strings
    if pd.isna(boxes_str) or boxes_str == "" or boxes_str == "nan":
        return mask

    try:
        boxes = ast.literal_eval(boxes_str)
    except:
        return mask

    if not boxes:
        return mask

    # Calculate scale factors
    scale_x = img_size / orig_w
    scale_y = img_size / orig_h

    for box in boxes:
        # Extract coordinates
        x = box["x"]
        y = box["y"]
        w = box["width"]
        h = box["height"]

        # Scale to new size
        x_min = int(x * scale_x)
        y_min = int(y * scale_y)
        x_max = int((x + w) * scale_x)
        y_max = int((y + h) * scale_y)

        # Clip to image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_size, x_max)
        y_max = min(img_size, y_max)

        # Draw filled rectangle
        mask[y_min:y_max, x_min:x_max] = 1.0

    return mask


def mask2boxes(mask, threshold=0.5):
    """
    Converts a probability mask into bounding boxes using contours.

    Args:
        mask (np.ndarray): Probability mask of shape (H, W).
        threshold (float): Threshold to binarize the mask.

    Returns:
        list: List of [confidence, x_min, y_min, x_max, y_max] relative to mask dimensions.
    """
    # Binarize
    bin_mask = (mask > threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        # Get bounding rect: x, y, w, h
        x, y, w, h = cv2.boundingRect(cnt)

        # Calculate confidence as mean probability within the contour/box
        # Using the bounding rect for mean calculation is faster and usually sufficient
        roi = mask[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        conf = np.mean(roi)

        # Format: x_min, y_min, x_max, y_max
        boxes.append([conf, x, y, x + w, y + h])

    return boxes


def format_submission(
    test_ids, study_preds, image_preds, save_path=Config.SUBMISSION_PATH
):
    """
    Formats predictions into the competition submission format.

    Args:
        test_ids (list): List of base IDs (without _study/_image suffix).
        study_preds (list/array): List of study class indices or probabilities.
                                  If probabilities, argmax is used.
                                  Shape: (N, 4) or (N,).
        image_preds (list): List of formatted image prediction strings.
                            Or list of list of boxes [conf, x1, y1, x2, y2].
                            Here we assume it's a list of strings or we format them.
                            To be robust, we'll assume `image_preds` is a list of
                            strings ready for the CSV, or we handle the logic here.

                            Let's implement the logic to take raw boxes and format them.
                            Assumption: image_preds is a list of lists of boxes
                            (scaled to original image size).
        save_path (str): Path to save the CSV.
    """

    rows = []

    for i, base_id in enumerate(test_ids):
        # --- Study Level ---
        # study_preds[i] should be the probabilities or class index
        # We need to output the label string.
        # Format: "class_name confidence 0 0 1 1"
        # We can predict multiple labels, but usually we pick the max.

        s_preds = study_preds[i]
        if hasattr(s_preds, "tolist"):
            s_preds = s_preds.tolist()

        # If s_preds is a vector of probabilities
        if isinstance(s_preds, list) or isinstance(s_preds, np.ndarray):
            # Get class with max confidence
            class_idx = np.argmax(s_preds)
            confidence = s_preds[class_idx]
            label = STUDY_CLASSES[class_idx]
            study_str = f"{label} {confidence:.6f} 0 0 1 1"
        else:
            # Fallback if just an index is passed
            label = STUDY_CLASSES[int(s_preds)]
            study_str = f"{label} 1 0 0 1 1"

        rows.append({"id": f"{base_id}_study", "PredictionString": study_str})

        # --- Image Level ---
        # image_preds[i] is expected to be a list of boxes: [conf, xmin, ymin, xmax, ymax]
        # or a pre-formatted string.

        i_preds = image_preds[i]

        if isinstance(i_preds, str):
            image_str = i_preds
        elif not i_preds:
            # No boxes predicted
            image_str = "none 1 0 0 1 1"
        else:
            # Format boxes
            box_strings = []
            for box in i_preds:
                # box: [conf, xmin, ymin, xmax, ymax]
                conf = box[0]
                xmin, ymin, xmax, ymax = box[1], box[2], box[3], box[4]
                box_strings.append(f"opacity {conf:.6f} {xmin} {ymin} {xmax} {ymax}")

            image_str = " ".join(box_strings)

        rows.append({"id": f"{base_id}_image", "PredictionString": image_str})

    df = pd.DataFrame(rows)

    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

    return df
