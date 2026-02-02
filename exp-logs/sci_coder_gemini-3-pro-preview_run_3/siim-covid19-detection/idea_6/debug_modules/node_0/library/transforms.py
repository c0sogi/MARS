import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(data: str):
    """
    Returns the albumentations transformation pipeline for the specified data split.

    Args:
        data (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The composed transformations.
    """

    # Common transformations: Letterbox Resizing and Normalization
    # 1. Resize longest side to target size
    # 2. Pad shorter side to target size (center padding, constant 0)
    # 3. Normalize
    # 4. Convert to Tensor

    if data == "train":
        return A.Compose(
            [
                # --- Letterbox Resizing ---
                A.LongestMaxSize(
                    max_size=Config.IMG_SIZE, interpolation=cv2.INTER_LINEAR
                ),
                A.PadIfNeeded(
                    min_height=Config.IMG_SIZE,
                    min_width=Config.IMG_SIZE,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
                # --- Augmentations ---
                # Random 90 degree rotation
                A.RandomRotate90(p=0.5 if Config.AUG_ROTATE_90 else 0.0),
                # Shift, Scale, and Rotate
                A.ShiftScaleRotate(
                    shift_limit=Config.AUG_SHIFT_LIMIT,
                    scale_limit=Config.AUG_SCALE_LIMIT,
                    rotate_limit=Config.AUG_ROTATE_LIMIT,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=Config.AUG_SHIFT_SCALE_ROTATE_PROB,
                ),
                # Cutout (CoarseDropout in newer albumentations)
                A.CoarseDropout(
                    max_holes=Config.AUG_CUTOUT_NUM_HOLES,
                    max_height=Config.AUG_CUTOUT_MAX_H_SIZE,
                    max_width=Config.AUG_CUTOUT_MAX_W_SIZE,
                    fill_value=0,
                    p=Config.AUG_CUTOUT_PROB,
                ),
                # --- Normalization & Tensor Conversion ---
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.0,  # Keep box even if partially occluded/out of bounds, filtering handled by dataset
            ),
        )

    elif data == "valid" or data == "test":
        return A.Compose(
            [
                # --- Letterbox Resizing ---
                A.LongestMaxSize(
                    max_size=Config.IMG_SIZE, interpolation=cv2.INTER_LINEAR
                ),
                A.PadIfNeeded(
                    min_height=Config.IMG_SIZE,
                    min_width=Config.IMG_SIZE,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
                # --- Normalization & Tensor Conversion ---
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"], min_visibility=0.0
            ),
        )

    else:
        raise ValueError(
            f"Unknown data split: {data}. Expected 'train', 'valid', or 'test'."
        )
