import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=480):
    """
    Returns the training transformations pipeline.

    Args:
        image_size (int): The target resolution for resizing. Defaults to 480.

    Returns:
        albumentations.Compose: The composition of transforms.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # Photometric distortions are excluded as per requirements
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_valid_transforms(image_size=480):
    """
    Returns the validation/test transformations pipeline.

    Args:
        image_size (int): The target resolution for resizing. Defaults to 480.

    Returns:
        albumentations.Compose: The composition of transforms.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


class CutMix:
    """
    Applies CutMix regularization to a batch of images and labels.
    This is intended to be used inside the training loop on GPU tensors.
    """

    def __init__(self, alpha=1.0):
        """
        Args:
            alpha (float): Parameter for the Beta distribution.
                           beta(alpha, alpha) is used to sample lambda.
        """
        self.alpha = alpha

    def __call__(self, data, target):
        """
        Applies CutMix to the batch.

        Args:
            data (torch.Tensor): Input batch of images (B, C, H, W).
            target (torch.Tensor): Input batch of labels (B, ...).

        Returns:
            torch.Tensor: Mixed images.
            torch.Tensor: Original targets (target_a).
            torch.Tensor: Shuffled targets (target_b).
            float: The mixing coefficient lambda (adjusted for exact box size).
        """
        indices = torch.randperm(data.size(0)).to(data.device)
        target_a = target
        target_b = target[indices]

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Generate random bounding box
        yl, yh, xl, xh = self.rand_bbox(data.size(), lam)

        # Adjust lambda to match the exact pixel ratio of the cropped area
        # lam is the proportion of the original image kept
        H = data.size(2)
        W = data.size(3)
        lam = 1 - ((yh - yl) * (xh - xl) / (H * W))

        # Paste the patch from the shuffled batch
        data[:, :, yl:yh, xl:xh] = data[indices, :, yl:yh, xl:xh]

        return data, target_a, target_b, lam

    @staticmethod
    def rand_bbox(size, lam):
        """
        Generates the coordinates for the random bounding box.

        Args:
            size (torch.Size): Size of the input batch (B, C, H, W).
            lam (float): The sampled mixing coefficient.

        Returns:
            tuple: (yl, yh, xl, xh) coordinates.
        """
        H = size[2]
        W = size[3]

        # Calculate cut ratio
        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)

        # Uniformly sample the center of the box
        cy = np.random.randint(H)
        cx = np.random.randint(W)

        # Calculate coordinates with clipping
        yl = np.clip(cy - cut_h // 2, 0, H)
        yh = np.clip(cy + cut_h // 2, 0, H)
        xl = np.clip(cx - cut_w // 2, 0, W)
        xh = np.clip(cx + cut_w // 2, 0, W)

        return yl, yh, xl, xh
