import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(split: str):
    """
    Constructs the data transformation pipeline for a specific dataset split.

    Args:
        split (str): The dataset split ('train', 'val', or 'test').

    Returns:
        A.Compose: The albumentations composition of transforms.
    """
    # Define Bounding Box Parameters
    # We use Pascal VOC format [x_min, y_min, x_max, y_max]
    # 'class_labels' ensures the label associated with each box is preserved
    bbox_params = A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_visibility=0.0,
        min_area=0.0,
    )

    # Base Geometric Normalization (Letterbox Resizing)
    # 1. Resize the longest dimension to the target size (Config.IMG_SIZE)
    # 2. Pad the shorter dimension to match the target size, resulting in a square image
    #    without distorting the aspect ratio.
    base_transforms = [
        A.LongestMaxSize(max_size=Config.IMG_SIZE, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=Config.IMG_SIZE,
            min_width=Config.IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
        ),
    ]

    # Normalization and Tensor Conversion
    # Using ImageNet statistics as the Swin Transformer backbone is pretrained on ImageNet.
    post_transforms = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    if split == "train":
        # Training Augmentations
        # 1. Multi-Scale Training: Simulate scale variations (+/- 20%) using ShiftScaleRotate.
        #    We disable shift and rotate to isolate the scaling effect.
        # 2. Random Horizontal Flip: Standard augmentation for chest X-rays.
        # 3. Random Brightness/Contrast: Improves robustness to lighting variations.
        aug_transforms = [
            A.ShiftScaleRotate(
                shift_limit=0.0,
                scale_limit=0.2,
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ]

        # Combine: Augment -> Letterbox Resize -> Normalize
        # Augmentations are applied before resizing to operate on the original resolution.
        transforms_list = aug_transforms + base_transforms + post_transforms
    else:
        # Validation/Test: Deterministic preprocessing
        transforms_list = base_transforms + post_transforms

    return A.Compose(transforms_list, bbox_params=bbox_params)
