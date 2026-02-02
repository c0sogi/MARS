import os
import cv2
import numpy as np
import pydicom
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def read_dicom_image(file_path):
    """
    Reads a DICOM file, handles PhotometricInterpretation, normalizes to 0-255,
    and converts to RGB.

    Args:
        file_path (str): Path to the DICOM file.

    Returns:
        np.ndarray: The image in RGB format (H, W, 3) with dtype uint8.
    """
    try:
        dcm = pydicom.dcmread(file_path)
        image = dcm.pixel_array

        # Handle Photometric Interpretation
        # MONOCHROME1: 0 is white, max is black. We want 0 to be black (air).
        # MONOCHROME2: 0 is black, max is white. This is what we want.
        photometric_interpretation = getattr(
            dcm, "PhotometricInterpretation", "MONOCHROME2"
        )
        if photometric_interpretation == "MONOCHROME1":
            image = np.max(image) - image

        # Normalize to 0-255
        if image.max() > 0:
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min())
            image = (image * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)

        # Convert to RGB (stacking channels)
        # ResNet expects 3 input channels
        image = np.stack([image, image, image], axis=-1)

        return image

    except Exception as e:
        print(f"Error reading DICOM {file_path}: {e}")
        # Return a black image of default size in case of error to prevent crash
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)


def get_transforms(data_split):
    """
    Returns albumentations transforms for the specified data split.

    Args:
        data_split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    shared_transforms = [
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(
            mean=Config.PIXEL_MEAN, std=Config.PIXEL_STD, max_pixel_value=255.0, p=1.0
        ),
        ToTensorV2(p=1.0),
    ]

    if data_split == "train":
        transforms = [
            A.HorizontalFlip(p=0.5),
            # Can add ShiftScaleRotate, RandomBrightnessContrast etc. here
            # Keeping it simple for baseline stability
        ] + shared_transforms
    else:
        transforms = shared_transforms

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def collate_fn(batch):
    """
    Custom collate function for object detection.
    Faster R-CNN expects a list of images and a list of target dictionaries.

    Args:
        batch: List of tuples (image, target, image_id) from Dataset.__getitem__

    Returns:
        tuple: (images, targets, image_ids)
    """
    images = []
    targets = []
    image_ids = []

    for b in batch:
        images.append(b[0])
        targets.append(b[1])
        image_ids.append(b[2])

    return images, targets, image_ids
