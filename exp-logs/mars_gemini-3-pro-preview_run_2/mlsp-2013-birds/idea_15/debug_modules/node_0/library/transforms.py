import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class TimeRolling(A.ImageOnlyTransform):
    """
    Custom Albumentations transform that performs a circular shift (roll)
    of the spectrogram along the time axis (x-axis).

    This exploits the temporal translation invariance of bird calls.
    """

    def __init__(self, always_apply=False, p=0.5):
        super(TimeRolling, self).__init__(always_apply, p)

    def apply(self, img, shift=0, **params):
        # img shape is (Height, Width) or (Height, Width, Channels)
        # Time axis is usually the width (axis 1)
        return np.roll(img, shift, axis=1)

    def get_params(self):
        return {
            "shift": np.random.randint(0, 1000)
        }  # Placeholder, actual shift depends on img width

    def get_transform_init_args_names(self):
        return ()

    def apply_to_image(self, img, **params):
        # Calculate shift dynamically based on current image width
        h, w = img.shape[:2]
        shift = np.random.randint(0, w)
        return self.apply(img, shift)


class SpecAugment(A.ImageOnlyTransform):
    """
    Implements SpecAugment: Frequency Masking and Time Masking.
    """

    def __init__(
        self,
        freq_mask_param=20,
        time_mask_param=40,
        num_freq_masks=1,
        num_time_masks=1,
        always_apply=False,
        p=0.5,
    ):
        super(SpecAugment, self).__init__(always_apply, p)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def apply(self, img, **params):
        # Work on a copy to avoid modifying original
        aug_img = img.copy()
        height, width = aug_img.shape[:2]

        # Frequency Masking
        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, height - f)
            aug_img[f0 : f0 + f, :] = 0

        # Time Masking
        for _ in range(self.num_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, width - t)
            aug_img[:, t0 : t0 + t] = 0

        return aug_img

    def get_transform_init_args_names(self):
        return (
            "freq_mask_param",
            "time_mask_param",
            "num_freq_masks",
            "num_time_masks",
        )


def to_3_channels(image, **kwargs):
    """
    Helper function to convert 1-channel grayscale images to 3-channel RGB.
    Required for models pretrained on ImageNet.
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def get_transforms(data_split, width, height):
    """
    Returns the Albumentations composition of transforms for a specific data split
    and target resolution.

    Args:
        data_split (str): 'train', 'val', or 'test'.
        width (int): Target width for resizing.
        height (int): Target height for resizing.

    Returns:
        A.Compose: The transform pipeline.
    """

    # Base transforms common to all splits (Resizing and Channel Conversion)
    # We resize first to ensure consistent dimensions for batching
    base_transforms = [
        A.Resize(height=height, width=width),
        A.Lambda(name="to_rgb", image=to_3_channels),
    ]

    # Normalization stats for ImageNet
    norm_transform = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # Tensor conversion
    to_tensor = ToTensorV2()

    if data_split == "train":
        # Training Pipeline with Augmentations
        return A.Compose(
            [
                *base_transforms,
                # 1. Spectrogram Time-Rolling (Circular Shift)
                # Directly addresses data scarcity by exploiting translation invariance
                TimeRolling(p=Config.TIME_ROLL_PROB),
                # 2. SpecAugment (Frequency and Time Masking)
                SpecAugment(
                    freq_mask_param=Config.FREQ_MASK_PARAM,
                    time_mask_param=Config.TIME_MASK_PARAM,
                    p=0.5,
                ),
                # 3. Random Brightness/Contrast
                # Simulates recording quality variations
                A.RandomBrightnessContrast(p=0.5),
                norm_transform,
                to_tensor,
            ]
        )

    elif data_split in ["val", "test", "valid"]:
        # Validation/Test Pipeline (Deterministic)
        return A.Compose([*base_transforms, norm_transform, to_tensor])

    else:
        raise ValueError(f"Unknown data_split: {data_split}")
