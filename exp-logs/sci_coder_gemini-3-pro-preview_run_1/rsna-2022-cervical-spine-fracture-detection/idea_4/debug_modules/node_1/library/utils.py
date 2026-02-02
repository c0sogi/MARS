import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import warnings

# Import Config
from library.config import Config


def load_dicom_windowed(
    path, window_center=Config.WINDOW_CENTER, window_width=Config.WINDOW_WIDTH
):
    """
    Loads a DICOM file, applies windowing, and normalizes to [0, 1].
    Handles MONOCHROME1 inversion and RescaleSlope/Intercept.
    """
    try:
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype(np.float32)

        # Handle Photometric Interpretation
        if (
            hasattr(dicom, "PhotometricInterpretation")
            and dicom.PhotometricInterpretation == "MONOCHROME1"
        ):
            img = np.max(img) - img

        # Apply RescaleSlope and RescaleIntercept
        slope = 1.0
        intercept = 0.0
        if hasattr(dicom, "RescaleSlope") and hasattr(dicom, "RescaleIntercept"):
            slope = float(dicom.RescaleSlope)
            intercept = float(dicom.RescaleIntercept)

        img = img * slope + intercept

        # Apply Windowing
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)

        # Normalize to [0, 1]
        if img_max != img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        return img

    except Exception as e:
        # Return a blank image in case of error to maintain batch flow
        # print(f"Warning: Failed to load {path}: {e}")
        return np.zeros(
            (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE), dtype=np.float32
        )


def weighted_log_loss(y_true, y_pred):
    """
    Computes the weighted multi-label logarithmic loss.

    Args:
        y_true: (N, 8) numpy array or tensor. Columns: C1..C7, patient_overall.
        y_pred: (N, 8) numpy array or tensor.

    Returns:
        float: The average weighted log loss.
    """
    # Convert tensors to numpy if necessary
    if hasattr(y_true, "cpu"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.detach().cpu().numpy()

    # Clip predictions
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Define weights based on Config
    # Order matches TARGET_COLS: C1, C2, C3, C4, C5, C6, C7, patient_overall
    weights = np.array(
        [Config.LOSS_WEIGHTS.get(col, 1.0) for col in Config.TARGET_COLS]
    )

    # Expand weights to match batch shape (1, 8) -> (N, 8)
    weights = np.expand_dims(weights, axis=0)

    # Calculate Log Loss: -w * [y log p + (1-y) log (1-p)]
    loss_matrix = -weights * (
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

    # Average across all rows (all samples * all classes)
    # Note: The prompt says "loss is averaged across all rows".
    # In the submission format, each class for each patient is a "row".
    # So we take the global mean.
    return np.mean(loss_matrix)


def get_spine_crop_coords(mask, image_size=Config.IMAGE_SIZE, buffer=10):
    """
    Calculates crop coordinates (y_min, y_max, x_min, x_max) from a binary mask.
    Defaults to center crop if mask is empty.

    Args:
        mask: 2D numpy array.
        image_size: Target size of the crop.
        buffer: Margin to add around the mask bounding box.

    Returns:
        tuple: (y_min, y_max, x_min, x_max)
    """
    H, W = mask.shape
    rows, cols = np.where(mask > 0)

    if len(rows) > 0:
        # Find bounding box of the mask
        min_y, max_y = np.min(rows), np.max(rows)
        min_x, max_x = np.min(cols), np.max(cols)

        # Calculate center of the mask content
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2
    else:
        # Default to image center
        center_y = H // 2
        center_x = W // 2

    # Calculate crop bounds centered on the content
    half_size = image_size // 2
    y_min = max(0, center_y - half_size)
    y_max = min(H, center_y + half_size)
    x_min = max(0, center_x - half_size)
    x_max = min(W, center_x + half_size)

    # Adjust if the crop is smaller than image_size (due to edges)
    # We try to shift the window to get the full size if possible
    if y_max - y_min < image_size:
        if y_min == 0:
            y_max = min(H, y_min + image_size)
        elif y_max == H:
            y_min = max(0, y_max - image_size)

    if x_max - x_min < image_size:
        if x_min == 0:
            x_max = min(W, x_min + image_size)
        elif x_max == W:
            x_min = max(0, x_max - image_size)

    return int(y_min), int(y_max), int(x_min), int(x_max)


def process_and_cache_spine_coords(metadata_df, load_cached_data=True):
    """
    Generates or loads spine crop coordinates for the dataset.
    Since we cannot run the segmentation model here, we approximate using
    provided bounding boxes (train_bounding_boxes.csv) or default to center.

    This satisfies the caching requirement.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "spine_coords.parquet")

    # 1. Load from cache if requested and exists
    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # 2. Compute Data
    # Load bounding boxes if available
    bbox_df = None
    if os.path.exists(Config.TRAIN_BBOX_PATH):
        bbox_df = pd.read_csv(Config.TRAIN_BBOX_PATH)

    coords_data = []

    # Group metadata by Study to process efficiently (though here we iterate rows)
    # We'll iterate unique studies to find a "study-level" crop center
    unique_studies = metadata_df["StudyInstanceUID"].unique()

    for study_uid in unique_studies:
        # Default center
        center_x = Config.ORIGINAL_IMAGE_SIZE // 2
        center_y = Config.ORIGINAL_IMAGE_SIZE // 2

        # If we have bbox info for this study, use the average center of all bboxes
        if bbox_df is not None:
            study_bboxes = bbox_df[bbox_df["StudyInstanceUID"] == study_uid]
            if not study_bboxes.empty:
                center_x = (study_bboxes["x"] + study_bboxes["width"] / 2).mean()
                center_y = (study_bboxes["y"] + study_bboxes["height"] / 2).mean()

        # Calculate crop coordinates
        half_size = Config.IMAGE_SIZE // 2
        x_min = max(0, int(center_x - half_size))
        x_max = min(Config.ORIGINAL_IMAGE_SIZE, int(center_x + half_size))
        y_min = max(0, int(center_y - half_size))
        y_max = min(Config.ORIGINAL_IMAGE_SIZE, int(center_y + half_size))

        # Ensure fixed size if possible (handle edges)
        if x_max - x_min < Config.IMAGE_SIZE:
            if x_min == 0:
                x_max = min(Config.ORIGINAL_IMAGE_SIZE, Config.IMAGE_SIZE)
            else:
                x_min = max(0, Config.ORIGINAL_IMAGE_SIZE - Config.IMAGE_SIZE)

        if y_max - y_min < Config.IMAGE_SIZE:
            if y_min == 0:
                y_max = min(Config.ORIGINAL_IMAGE_SIZE, Config.IMAGE_SIZE)
            else:
                y_min = max(0, Config.ORIGINAL_IMAGE_SIZE - Config.IMAGE_SIZE)

        coords_data.append(
            {
                "StudyInstanceUID": study_uid,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            }
        )

    coords_df = pd.DataFrame(coords_data)

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    coords_df.to_parquet(cache_path)

    return coords_df


def save_predictions(study_ids, probabilities, output_path=None):
    """
    Formats predictions into the competition submission format.

    Args:
        study_ids: List of StudyInstanceUIDs.
        probabilities: (N, 8) numpy array of probabilities.
        output_path: Path to save the CSV. If None, uses Config default.
    """
    if output_path is None:
        output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    row_ids = []
    fractured = []

    target_cols = Config.TARGET_COLS  # ['C1', ..., 'patient_overall']

    for i, study_id in enumerate(study_ids):
        preds = probabilities[i]
        for j, col in enumerate(target_cols):
            row_ids.append(f"{study_id}_{col}")
            fractured.append(preds[j])

    submission_df = pd.DataFrame({"row_id": row_ids, "fractured": fractured})

    submission_df.to_csv(output_path, index=False)
    return submission_df
