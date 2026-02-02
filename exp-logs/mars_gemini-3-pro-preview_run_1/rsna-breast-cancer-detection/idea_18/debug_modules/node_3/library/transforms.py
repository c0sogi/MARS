import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class PairedAugmentation:
    """
    Handles synchronized geometric augmentations for the Pyramid Symmetry-Difference Siamese Network.

    Ensures that the Target Image and Contralateral Image undergo the exact same
    spatial transformations (Flip, Rotate, Shift) to maintain pixel-wise correspondence
    for the difference module.
    """

    def __init__(self):
        self.height, self.width = Config.IMAGE_SIZE

        # ---------------------------------------------------------------------
        # Training Pipeline (Synchronized Geometric Transforms)
        # ---------------------------------------------------------------------
        # We use 'additional_targets' to bind 'image_contra' to the same params as 'image'.
        # Photometric augmentations (Brightness, Contrast) are DISABLED as per instructions.
        self.transform_train = A.Compose(
            [
                A.Resize(self.height, self.width),
                # Random Flips (p=0.5)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Affine Transforms (Shift, Scale, Rotate)
                # Border mode constant (0) ensures we don't introduce artifacts at edges
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Normalize to center the data (assuming [0, 1] input from _preprocess)
                # Mean/Std 0.5 maps [0, 1] -> [-1, 1]
                A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0),
                # Convert to PyTorch Tensor (C, H, W)
                ToTensorV2(),
            ],
            additional_targets={"image_contra": "image"},
        )

        # ---------------------------------------------------------------------
        # Validation/Test Pipeline (Deterministic)
        # ---------------------------------------------------------------------
        self.transform_val = A.Compose(
            [
                A.Resize(self.height, self.width),
                A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0),
                ToTensorV2(),
            ],
            additional_targets={"image_contra": "image"},
        )

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Standardizes input images to float32 [0, 1] range and ensures HWC format.
        Handles both 8-bit and 16-bit (DICOM) inputs robustly.
        """
        # Convert to float32 for precision
        img = img.astype(np.float32)

        # Robust Min-Max Scaling
        # This preserves relative density patterns within the image while
        # mapping to a standard range for the neural network.
        min_val = img.min()
        max_val = img.max()

        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            # Handle blank/constant images
            img = np.zeros_like(img)

        # Ensure Channel Dimension exists (H, W) -> (H, W, 1)
        # Albumentations expects HWC format
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]

        return img

    def __call__(
        self, image_target: np.ndarray, image_contra: np.ndarray, mode: str = "train"
    ):
        """
        Applies the transformation pipeline to a pair of images.

        Args:
            image_target (np.ndarray): The primary image (candidate for cancer).
            image_contra (np.ndarray): The contralateral image (opposing breast).
            mode (str): 'train' for augmentation, 'val' or 'test' for deterministic.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The transformed image tensors (C, H, W).
        """
        # 1. Preprocess (Scale to [0, 1] & Reshape)
        image_target = self._preprocess(image_target)
        image_contra = self._preprocess(image_contra)

        # 2. Apply Pipeline
        if mode == "train":
            augmented = self.transform_train(
                image=image_target, image_contra=image_contra
            )
        else:
            augmented = self.transform_val(
                image=image_target, image_contra=image_contra
            )

        # 3. Return Tensors
        return augmented["image"], augmented["image_contra"]
