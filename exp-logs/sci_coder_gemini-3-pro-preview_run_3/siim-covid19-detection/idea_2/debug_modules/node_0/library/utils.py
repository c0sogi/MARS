import os
import ast
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from library.config import Config


def read_dicom(file_path, image_size=None, fix_monochrome=True):
    """
    Reads a DICOM file, handles monochrome interpretation, normalizes to 0-255, and resizes.

    Args:
        file_path (str): Relative path to the DICOM file from Config.INPUT_DIR.
        image_size (int or tuple, optional): Target size (H, W) or int for square.
        fix_monochrome (bool): Whether to fix MONOCHROME1 interpretation.

    Returns:
        np.ndarray: The processed image (uint8).
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"DICOM file not found: {full_path}")

    # Read DICOM
    dcm = pydicom.dcmread(full_path)
    img = dcm.pixel_array.astype(np.float32)

    # Handle Photometric Interpretation
    # If MONOCHROME1, 0 is white and max is black. We want 0=black, max=white (MONOCHROME2 style).
    if (
        fix_monochrome
        and hasattr(dcm, "PhotometricInterpretation")
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

    # Resize
    if image_size is not None:
        if isinstance(image_size, int):
            target_size = (image_size, image_size)
        else:
            target_size = image_size
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    return img


def load_dataset_dataframe(split="train", load_cached_data=True):
    """
    Loads the metadata dataframe for a specific split with caching.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe with 'boxes' parsed and 'class_id' mapped.
    """
    # Determine paths
    if split == "train":
        csv_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.CACHED_TRAIN_DF_PATH
    elif split == "val":
        csv_path = Config.VAL_METADATA_PATH
        cache_path = Config.CACHED_VAL_DF_PATH
    elif split == "test":
        csv_path = Config.TEST_METADATA_PATH
        cache_path = Config.CACHED_TEST_DF_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Parse 'boxes' back to list if they are strings
            if "boxes" in df.columns:
                # Check first non-null element to see if it's a string
                sample = (
                    df["boxes"].dropna().iloc[0]
                    if not df["boxes"].dropna().empty
                    else None
                )
                if isinstance(sample, str):
                    df["boxes"] = df["boxes"].apply(
                        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
                    )
            return df
        except Exception as e:
            print(f"Failed to load cache for {split}: {e}. Recomputing...")

    # 2. Compute from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Process 'boxes'
    if "boxes" in df.columns:
        df["boxes"] = df["boxes"].fillna("[]")
        df["boxes"] = df["boxes"].apply(ast.literal_eval)

    # Process Class Labels (for train/val)
    # Map one-hot columns to single class_id
    if "Typical Appearance" in df.columns:

        def get_class_id(row):
            if row["Typical Appearance"]:
                return Config.CLASS_MAPPING["Typical Appearance"]
            if row["Indeterminate Appearance"]:
                return Config.CLASS_MAPPING["Indeterminate Appearance"]
            if row["Atypical Appearance"]:
                return Config.CLASS_MAPPING["Atypical Appearance"]
            return 0  # Negative for Pneumonia

        df["class_id"] = df.apply(get_class_id, axis=1)

    # 3. Save to cache
    # Convert 'boxes' to string for safe parquet storage
    df_to_save = df.copy()
    if "boxes" in df_to_save.columns:
        df_to_save["boxes"] = df_to_save["boxes"].apply(str)

    try:
        df_to_save.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df


def collate_fn(batch):
    """
    Collate function for PyTorch DataLoader.

    Args:
        batch: List of tuples (image, target, image_id).
               image: torch.Tensor
               target: dict or list
               image_id: str

    Returns:
        images: torch.Tensor (stacked)
        targets: List of targets
        image_ids: List of image IDs
    """
    images = []
    targets = []
    image_ids = []

    for img, target, img_id in batch:
        images.append(img)
        targets.append(target)
        image_ids.append(img_id)

    # Stack images into a batch tensor
    images = torch.stack(images, dim=0)

    return images, targets, image_ids


def format_image_prediction_string(boxes, scores):
    """
    Formats the prediction string for an image.

    Args:
        boxes (list/array): List of [xmin, ymin, xmax, ymax].
        scores (list/array): List of confidence scores.

    Returns:
        str: Formatted prediction string (e.g., "opacity 0.5 100 100 200 200 ...").
    """
    if len(boxes) == 0:
        return Config.NONE_PREDICTION

    pred_strings = []
    for box, score in zip(boxes, scores):
        xmin, ymin, xmax, ymax = box
        pred_strings.append(
            f"{Config.OPACITY_LABEL} {score:.6f} {xmin} {ymin} {xmax} {ymax}"
        )

    return " ".join(pred_strings)


def format_study_prediction_string(boxes, scores, class_ids):
    """
    Formats the prediction string for a study.

    Args:
        boxes (list/array): List of detected boxes.
        scores (list/array): List of confidence scores.
        class_ids (list/array): List of predicted class IDs for the boxes.

    Returns:
        str: Formatted prediction string (e.g., "typical 0.8 0 0 1 1").
    """
    if len(boxes) == 0:
        return f"{Config.NEGATIVE_STUDY_LABEL} 1 0 0 1 1"

    # Identify the box with the highest confidence
    max_idx = np.argmax(scores)
    best_class_id = class_ids[max_idx]
    best_score = scores[max_idx]

    # Map class ID to label
    if best_class_id in Config.ID_TO_SUBMISSION_STRING:
        label = Config.ID_TO_SUBMISSION_STRING[best_class_id]
    else:
        # Fallback if class_id is 0 or unknown, though logic should prevent this for detections
        label = Config.NEGATIVE_STUDY_LABEL

    return f"{label} {best_score:.6f} 0 0 1 1"
