import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from library.config import Config


class VolumetricReplayWrapper:
    """
    Wraps an Albumentations pipeline to ensure consistent augmentation parameters
    across a sequence of slices (Volumetric Consistency).

    This is critical for 2.5D/3D inputs where geometric transformations (like rotation)
    must be applied identically to every slice in the stack to preserve physical coherence.
    """

    def __init__(self, transforms: list):
        self.transforms = A.ReplayCompose(transforms)

    def __call__(self, volume: np.ndarray) -> torch.Tensor:
        """
        Args:
            volume (np.ndarray): Input volume of shape (Seq, H, W, C).
                                 Dtype should be uint8 (0-255).
        Returns:
            torch.Tensor: Transformed volume of shape (Seq, C, H, W).
        """
        if len(volume) == 0:
            return torch.empty(0)

        # Apply transforms to the first slice to generate and lock in the random parameters.
        # Albumentations expects 'image' kwarg.
        # res['image'] will be a Tensor (C, H, W) due to ToTensorV2 in the pipeline.
        res = self.transforms(image=volume[0])
        replay_params = res["replay"]

        output_slices = [res["image"]]

        # Replay the exact same transforms for the remaining slices in the sequence
        for i in range(1, len(volume)):
            res_i = A.ReplayCompose.replay(replay_params, image=volume[i])
            output_slices.append(res_i["image"])

        # Stack along the sequence dimension
        # Result shape: (Seq, C, H, W)
        return torch.stack(output_slices)


def get_transforms(phase: str = "train"):
    """
    Returns a VolumetricReplayWrapper containing the augmentation pipeline.

    Args:
        phase (str): 'train' or 'valid' (also used for test).
    """
    height, width = Config.IMAGE_SIZE

    # Base transforms applied to all phases
    # Resize is added as a safety measure, though data loader should handle it.
    common_transforms = [
        A.Resize(height=height, width=width, always_apply=True),
        A.Normalize(mean=Config.MEAN, std=Config.STD, always_apply=True),
        ToTensorV2(),
    ]

    if phase == "train":
        # Train pipeline: Geometric augmentations + Common
        # We use ShiftScaleRotate to fuse affine transforms as per the design idea.
        # No Cutout or Dropout is used on the input.
        train_transforms = [
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,  # Constant padding
                value=0,  # Pad with black
                p=0.5,
            )
        ] + common_transforms
        return VolumetricReplayWrapper(train_transforms)

    else:
        # Valid/Test pipeline: Common only (Deterministic)
        return VolumetricReplayWrapper(common_transforms)
