import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List
from PIL import Image

# Import configuration
from library.config import Config


class LeafImageProcessor:
    """
    A processor for loading leaf images and generating multi-view augmentations
    for Manifold-Densified classification.
    """

    def __init__(self):
        """
        Initializes the image processor with the standard transformation pipeline
        required for pre-trained models (DINOv2, ConvNeXt).
        """
        # Standard ImageNet normalization statistics
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Transformation pipeline:
        # 1. Convert numpy array to PIL Image
        # 2. Resize to the target input size (e.g., 224x224)
        # 3. Convert to Tensor (scales pixels to [0, 1])
        # 4. Normalize with ImageNet mean/std
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def load_image(self, rel_path: str) -> np.ndarray:
        """
        Loads an image from the disk, ensuring it is in RGB format.

        Args:
            rel_path (str): The relative path to the image file (e.g., 'images/10.jpg').

        Returns:
            np.ndarray: The loaded image as a numpy array in RGB format.
        """
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Verify file existence
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found at: {full_path}")

        # Load image using OpenCV
        img = cv2.imread(full_path)
        if img is None:
            raise ValueError(f"Failed to read image file: {full_path}")

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _pad_to_square(self, image: np.ndarray) -> np.ndarray:
        """
        Helper method to pad a rectangular image to a square shape using white padding.
        This prevents content clipping during rotation.

        Args:
            image (np.ndarray): Input RGB image.

        Returns:
            np.ndarray: Square padded image.
        """
        h, w = image.shape[:2]

        # If already square, return as is
        if h == w:
            return image

        # Determine the size of the square canvas
        size = max(h, w)

        # Calculate padding amounts
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left

        # Pad with white (255, 255, 255) since background is white
        padded_image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

        return padded_image

    def generate_rotated_views(self, image: np.ndarray) -> List[torch.Tensor]:
        """
        Generates a list of rotated views for the given image based on the
        angles defined in Config.ROTATION_ANGLES.

        Process:
        1. Pad image to square.
        2. Rotate image around the center.
        3. Apply resizing and normalization.

        Args:
            image (np.ndarray): Input RGB image.

        Returns:
            List[torch.Tensor]: A list of tensors corresponding to each rotation angle.
                                Each tensor has shape (3, Config.IMG_SIZE, Config.IMG_SIZE).
        """
        # Step 1: Pad to square to ensure rotation doesn't crop the leaf
        square_img = self._pad_to_square(image)

        h, w = square_img.shape[:2]
        center = (w // 2, h // 2)

        views = []

        # Step 2: Iterate through defined rotation angles
        for angle in Config.ROTATION_ANGLES:
            # Compute rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Perform affine transformation (Rotation)
            # Fill new border areas with white (255, 255, 255)
            rotated_img = cv2.warpAffine(
                square_img,
                M,
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

            # Step 3: Apply transforms (Resize -> ToTensor -> Normalize)
            img_tensor = self.transform(rotated_img)
            views.append(img_tensor)

        return views
