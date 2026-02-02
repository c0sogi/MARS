import os
import numpy as np
import pandas as pd
import cv2
import torch
import pydicom
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config
from library.utils import get_logger

# Initialize Logger
logger = get_logger("data_processing")


def load_dicom(path):
    """
    Loads a DICOM file, handles photometric interpretation, and converts to 8-bit uint.

    Args:
        path (str): Path to the .dcm file.

    Returns:
        np.ndarray: 2D image array (uint8).
    """
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array

        # Handle Photometric Interpretation
        # MONOCHROME1: 0 is white (dense), so we invert it to match MONOCHROME2 (0 is black)
        if (
            hasattr(ds, "PhotometricInterpretation")
            and ds.PhotometricInterpretation == "MONOCHROME1"
        ):
            img = np.max(img) - img

        # Normalize to 0-255 range
        if img.max() > 0:
            img = (img - img.min()) / (img.max() - img.min()) * 255.0

        return img.astype(np.uint8)

    except Exception as e:
        # In case of failure (e.g., missing JPEG2000 drivers), return a blank image
        # This ensures the pipeline doesn't crash for a single bad file
        # logger.warning(f"Error loading {path}: {e}")
        return np.zeros((Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]), dtype=np.uint8)


def crop_breast_roi(image):
    """
    Crops the breast region of interest (ROI) using Otsu's thresholding.

    Args:
        image (np.ndarray): Input grayscale image (uint8).

    Returns:
        np.ndarray: Cropped image containing the breast tissue.
    """
    # Binarize the image using Otsu's thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assume the largest contour is the breast
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # Crop the image
        return image[y : y + h, x : x + w]
    else:
        # Fallback if no contours found
        return image


def resize_and_normalize(image):
    """
    Resizes the image to Config dimensions, converts to Tensor, and normalizes.

    Args:
        image (np.ndarray): Input image (uint8).

    Returns:
        torch.Tensor: Normalized image tensor of shape (3, H, W).
    """
    # Resize
    # cv2.resize expects (Width, Height)
    target_h, target_w = Config.IMAGE_SIZE
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Convert to RGB (3 channels) for standard backbone compatibility
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # To Tensor (C, H, W) and scale to [0, 1]
    image_tensor = transforms.functional.to_tensor(image)

    # Normalize with ImageNet mean/std
    normalize = transforms.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)
    image_tensor = normalize(image_tensor)

    return image_tensor


def process_and_cache_data(
    metadata_path, output_path, load_cached_data=True, is_test=False
):
    """
    Groups metadata by patient and laterality to form bags, and caches the result.

    Args:
        metadata_path (str): Path to the source CSV.
        output_path (str): Path to save/load the Parquet cache.
        load_cached_data (bool): Whether to attempt loading from cache.
        is_test (bool): Whether processing test data (affects target processing).

    Returns:
        pd.DataFrame: The processed bag-level dataframe.
    """
    # 1. Try to load cache
    if load_cached_data and os.path.exists(output_path):
        logger.info(f"Loading cached bag data from {output_path}")
        try:
            return pd.read_parquet(output_path)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Re-processing.")

    # 2. Process from scratch
    logger.info(f"Processing metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)

    # Create Bag ID
    # For test, prediction_id is already unique per breast (e.g., 10116_L)
    # For train, we construct it: patient_id + laterality
    if "prediction_id" not in df.columns:
        df["bag_id"] = df["patient_id"].astype(str) + "_" + df["laterality"]
    else:
        df["bag_id"] = df["prediction_id"]

    # Define Aggregations
    # We collect all file paths belonging to the bag
    agg_dict = {
        "file_path": list,
        "image_id": list,
        "patient_id": "first",
        "laterality": "first",
    }

    if not is_test:
        # Targets
        # Cancer: Max over the bag (if any view is cancer, the breast is cancer)
        agg_dict["cancer"] = "max"

        # Density: Usually consistent per patient/breast. Take first.
        if "density" in df.columns:
            agg_dict["density"] = "first"

        # Biopsy: Max over bag
        if "biopsy" in df.columns:
            agg_dict["biopsy"] = "max"

    # Group by Bag ID
    df_bag = df.groupby("bag_id").agg(agg_dict).reset_index()

    # Process Targets for Training
    if not is_test:
        # Map Density (A->0, B->1, C->2, D->3, Missing->-1)
        if "density" in df_bag.columns:
            density_map = {"A": 0, "B": 1, "C": 2, "D": 3}
            # Handle NaNs by filling with a placeholder (e.g., -1)
            df_bag["density_label"] = (
                df_bag["density"].map(density_map).fillna(-1).astype(int)
            )

        # Handle Biopsy NaNs (assume 0 if missing, though typically not missing in this dataset)
        if "biopsy" in df_bag.columns:
            df_bag["biopsy"] = df_bag["biopsy"].fillna(0).astype(int)

    # Save to Cache
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_bag.to_parquet(output_path)
    logger.info(f"Saved processed bag cache to {output_path}")

    return df_bag


class BreastCancerDataset(Dataset):
    """
    PyTorch Dataset for Multi-View Breast Cancer Detection.
    Loads a bag of images for a specific breast (patient + laterality).
    """

    def __init__(self, df, input_dir=Config.INPUT_DIR, is_train=True):
        """
        Args:
            df (pd.DataFrame): Bag-level dataframe.
            input_dir (str): Root directory for images.
            is_train (bool): Whether this is a training dataset (returns targets).
        """
        self.df = df
        self.input_dir = input_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Images
        file_paths = row["file_path"]
        images = []

        for rel_path in file_paths:
            full_path = os.path.join(self.input_dir, rel_path)

            # Load
            img_arr = load_dicom(full_path)

            # Crop
            img_arr = crop_breast_roi(img_arr)

            # Resize & Normalize (returns Tensor)
            img_tensor = resize_and_normalize(img_arr)
            images.append(img_tensor)

        # Stack images: (Num_Views, C, H, W)
        if len(images) > 0:
            images_stack = torch.stack(images)
        else:
            # Fallback for empty bag (should not happen based on metadata)
            # Create a blank tensor
            images_stack = torch.zeros(
                (1, 3, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])
            )

        # 2. Prepare Targets
        if self.is_train:
            targets = {
                "cancer": torch.tensor(row["cancer"], dtype=torch.float32),
                "density": (
                    torch.tensor(row["density_label"], dtype=torch.long)
                    if "density_label" in row
                    else torch.tensor(-1, dtype=torch.long)
                ),
                "biopsy": (
                    torch.tensor(row["biopsy"], dtype=torch.float32)
                    if "biopsy" in row
                    else torch.tensor(0.0, dtype=torch.float32)
                ),
            }
            return images_stack, targets
        else:
            # For inference, we need the prediction_id
            return images_stack, row["bag_id"]


def collate_bag_fn(batch):
    """
    Custom collate function to handle variable number of images per bag.
    """
    # batch is a list of tuples: (images_stack, targets/id)
    images_list = []
    targets_list = []

    for img_stack, target in batch:
        images_list.append(img_stack)
        targets_list.append(target)

    # We cannot stack images_list directly because dim 0 (num_views) varies.
    # We return a list of tensors for images, and stacked tensors for targets (if train).

    if isinstance(targets_list[0], dict):
        # Training mode: stack targets
        collated_targets = {
            key: torch.stack([t[key] for t in targets_list])
            for key in targets_list[0].keys()
        }
        return images_list, collated_targets
    else:
        # Inference mode: return list of IDs
        return images_list, targets_list
