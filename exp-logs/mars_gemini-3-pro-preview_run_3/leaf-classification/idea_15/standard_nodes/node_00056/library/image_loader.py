import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config


class LeafDataset(Dataset):
    """
    A PyTorch Dataset class designed for the Hyper-Densified Leaf Classification task.

    It handles:
    1. Loading binary leaf images.
    2. Applying deterministic rotations for multi-view feature extraction.
    3. Resizing and formatting images for DINOv2/ConvNeXt input.
    4. Normalizing using ImageNet statistics.
    """

    def __init__(self, file_paths, rotation_angle=0):
        """
        Initialize the dataset.

        Args:
            file_paths (list): List of absolute or relative file paths to the images.
            rotation_angle (float): The angle in degrees to rotate the image (counter-clockwise).
                                    Used to generate specific views (e.g., 0, 10, ..., 350).
        """
        self.file_paths = file_paths
        self.rotation_angle = rotation_angle

        # Standard ImageNet normalization is required for both DINOv2 and ConvNeXt
        # We compose ToTensor (which scales 0-255 to 0.0-1.0) with Normalize
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Loads, transforms, and returns the image at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            torch.Tensor: Preprocessed image tensor of shape (3, H, W).
        """
        img_path = self.file_paths[idx]

        # Load image in grayscale. The dataset consists of binary images.
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Robustness check: if image fails to load, return a blank white image
        # This ensures the pipeline doesn't crash on a single bad file
        if img is None:
            # 255 represents white background
            img = np.full((Config.IMG_SIZE, Config.IMG_SIZE), 255, dtype=np.uint8)

        # Apply rotation if a specific angle is requested
        if self.rotation_angle != 0:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)

            # Get rotation matrix for the specified angle
            # 1.0 is the scale factor (no scaling)
            M = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)

            # Warp the image. borderValue=255 ensures that any new background
            # introduced by rotation is white, matching the original background.
            img = cv2.warpAffine(img, M, (w, h), borderValue=255)

        # Resize to the input size expected by the models (224x224)
        # INTER_AREA is preferred for downsampling to prevent aliasing artifacts
        if (img.shape[0] != Config.IMG_SIZE) or (img.shape[1] != Config.IMG_SIZE):
            img = cv2.resize(
                img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
            )

        # Convert Grayscale to RGB
        # Although the image is binary, the pre-trained backbones expect 3 channels.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Apply ToTensor and Normalize
        img_tensor = self.transform(img_rgb)

        return img_tensor
