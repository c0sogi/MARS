import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class VolumetricTransforms:
    """
    Applies volumetric-consistent augmentations to a sequence of 2.5D slices.

    Crucially, this class ensures that geometric transformations (like rotation,
    shifting, and scaling) are applied identically across all slices in a scan.
    This preserves the physical alignment of the spine in the Z-axis, which is
    required for the Multi-Scale Contextualized module to learn valid spatial features.
    """

    def __init__(self, phase: str = "train"):
        self.phase = phase
        self.image_size = Config.IMAGE_SIZE

        if self.phase == "train":
            # ReplayCompose allows us to record the parameters (e.g., rotation angle)
            # applied to the first image and replay them on subsequent images.
            self.transform = A.ReplayCompose(
                [
                    A.Resize(height=self.image_size, width=self.image_size),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=15,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.5,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.3
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            # For validation and test, we only resize and normalize.
            # No random geometric transforms, so ReplayCompose is not needed.
            self.transform = A.Compose(
                [
                    A.Resize(height=self.image_size, width=self.image_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def __call__(self, volume: np.ndarray) -> torch.Tensor:
        """
        Apply transforms to a 2.5D volume.

        Args:
            volume (np.ndarray): Input volume of shape (Depth, Height, Width, Channels).
                                 Typically (64, H, W, 3).

        Returns:
            torch.Tensor: Transformed volume of shape (Depth, Channels, Height, Width).
        """
        depth = volume.shape[0]
        transformed_slices = []

        if self.phase == "train":
            # 1. Apply transform to the first slice and capture the parameters
            # volume[0] is (H, W, C)
            res = self.transform(image=volume[0])
            transformed_slices.append(res["image"])
            replay_params = res["replay"]

            # 2. Replay the exact same transform for the remaining slices
            # This ensures the "Volumetric-Consistent Augmentation"
            for i in range(1, depth):
                # Use the instance method .replay() to apply saved params
                res_i = self.transform.replay(replay_params, image=volume[i])
                transformed_slices.append(res_i["image"])

        else:
            # Independent application for deterministic transforms
            for i in range(depth):
                res = self.transform(image=volume[i])
                transformed_slices.append(res["image"])

        # Stack list of tensors into a single 4D tensor
        # List of (C, H, W) -> (D, C, H, W)
        return torch.stack(transformed_slices)


def get_transforms(phase: str = "train"):
    """
    Factory function to retrieve the appropriate transform pipeline.

    Args:
        phase (str): 'train' for augmentation, 'valid' or 'test' for deterministic preprocessing.
    """
    return VolumetricTransforms(phase)
