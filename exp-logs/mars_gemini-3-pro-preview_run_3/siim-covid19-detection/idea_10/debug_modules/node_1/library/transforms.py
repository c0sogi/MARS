import cv2
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class LargeScaleJitterCompose:
    """
    Applies Large Scale Jittering (LSJ) for training.

    Strategy:
    1. Sample a random scale factor.
    2. Resize the image such that the longest side equals (image_size * scale).
       This preserves the aspect ratio.
    3. Pad the image if the resized dimensions are smaller than the crop size.
    4. Randomly crop a fixed-size patch (image_size x image_size).

    This simulates seeing the object at various scales (zoomed in or out)
    while maintaining the fixed input size required by the network.
    """

    def __init__(self, image_size, scale_limit=(0.1, 2.0), p_flip=0.5):
        self.image_size = image_size
        self.scale_limit = scale_limit
        self.p_flip = p_flip

        # Static transforms that don't depend on the random scale
        self.flip = A.HorizontalFlip(p=p_flip)
        self.normalize = A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        self.to_tensor = ToTensorV2()

    def __call__(self, image, bboxes=None, class_labels=None):
        # Ensure image has 3 channels (H, W) -> (H, W, 3)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Handle case where bboxes are not provided (e.g. inference)
        if bboxes is None:
            bboxes = []
        if class_labels is None:
            class_labels = []

        # 1. Sample random scale
        scale = random.uniform(self.scale_limit[0], self.scale_limit[1])
        target_dim = int(self.image_size * scale)

        # 2. Construct dynamic pipeline
        # LongestMaxSize: Resizes image so the longest edge is equal to target_dim
        # PadIfNeeded: Ensures the image is at least image_size x image_size.
        #              If target_dim < image_size, this pads the small image.
        # RandomCrop: Extracts the final input patch.
        #             If target_dim > image_size, this crops a region.
        #             If target_dim < image_size, this effectively takes the whole padded image.

        transforms = [
            A.LongestMaxSize(max_size=target_dim, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=self.image_size,
                min_width=self.image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            A.RandomCrop(height=self.image_size, width=self.image_size, p=1.0),
            self.flip,
            self.normalize,
            self.to_tensor,
        ]

        # Define BboxParams
        # min_visibility=0.1 ensures we drop boxes that are mostly cut off by the crop
        bbox_params = A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"], min_visibility=0.1
        )

        comp = A.Compose(transforms, bbox_params=bbox_params)

        return comp(image=image, bboxes=bboxes, class_labels=class_labels)


class ValidTransformCompose:
    """
    Applies deterministic Letterbox Resizing for validation and testing.

    Strategy:
    1. Resize the image so the longest side equals image_size.
    2. Pad the shorter side symmetrically to create a square image_size x image_size.

    This preserves the aspect ratio of the original radiograph, preventing distortion
    of opacities.
    """

    def __init__(self, image_size):
        self.image_size = image_size

        self.transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(
                    min_height=image_size,
                    min_width=image_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"]
            ),
        )

    def __call__(self, image, bboxes=None, class_labels=None):
        # Ensure image has 3 channels
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Handle missing boxes/labels for inference
        if bboxes is None:
            bboxes = []
        if class_labels is None:
            class_labels = []

        return self.transforms(image=image, bboxes=bboxes, class_labels=class_labels)


def get_transforms(data="train"):
    """
    Factory function to retrieve the appropriate transformation pipeline.

    Args:
        data (str): One of 'train', 'valid', or 'test'.

    Returns:
        Callable: A transform instance that accepts (image, bboxes, class_labels).
    """
    cfg = Config.get_transforms_config()
    image_size = cfg["image_size"]

    if data == "train":
        return LargeScaleJitterCompose(
            image_size=image_size, scale_limit=cfg["scale_limit"], p_flip=cfg["p_flip"]
        )
    elif data == "valid" or data == "test":
        return ValidTransformCompose(image_size=image_size)
    else:
        raise ValueError(f"Unknown data mode: {data}")
