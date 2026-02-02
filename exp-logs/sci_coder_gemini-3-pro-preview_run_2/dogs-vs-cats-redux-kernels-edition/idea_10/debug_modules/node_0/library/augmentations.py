import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_train_transforms(
    image_size=Config.IMAGE_SIZE,
    min_scale=Config.RRC_MIN_SCALE,
    max_scale=Config.RRC_MAX_SCALE,
):
    """
    Returns the training image transformations.

    Args:
        image_size (int): Target image size (height and width).
        min_scale (float): Minimum scale for RandomResizedCrop.
        max_scale (float): Maximum scale for RandomResizedCrop.

    Returns:
        A.Compose: Albumentations composition of transforms.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(
                height=image_size, width=image_size, scale=(min_scale, max_scale), p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_valid_transforms(image_size=Config.IMAGE_SIZE):
    """
    Returns the validation/test image transformations.

    Args:
        image_size (int): Target image size (height and width).

    Returns:
        A.Compose: Albumentations composition of transforms.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class MixupCutmixCollator:
    """
    Custom Collator that applies Mixup or CutMix to a batch of data.
    """

    def __init__(
        self,
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        mixup_prob=Config.MIXUP_PROB,
        cutmix_prob=Config.CUTMIX_PROB,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob

    def __call__(self, batch):
        """
        Processes a batch of data, applying Mixup or CutMix stochastically.

        Args:
            batch: List of tuples (image, label).

        Returns:
            batch_x: Tensor of images (B, C, H, W).
            batch_y: Tensor of labels (B).
        """
        # Separate images and labels
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Stack images into a tensor
        batch_x = torch.stack(images)

        # Convert labels to float tensor for soft label mixing
        batch_y = torch.tensor(labels, dtype=torch.float32)

        # Stochastic selection of augmentation
        # We assume mixup_prob + cutmix_prob <= 1.0
        choice = np.random.rand()

        if choice < self.mixup_prob:
            # Apply Mixup
            batch_x, batch_y = self.mixup(batch_x, batch_y)
        elif choice < self.mixup_prob + self.cutmix_prob:
            # Apply CutMix
            batch_x, batch_y = self.cutmix(batch_x, batch_y)

        # If neither is selected (if probs sum < 1.0), return original batch
        return batch_x, batch_y

    def mixup(self, x, y):
        """Applies Mixup augmentation."""
        # Sample lambda from Beta distribution
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        # Mix images
        mixed_x = lam * x + (1 - lam) * x[index, :]

        # Mix labels
        y_a, y_b = y, y[index]
        mixed_y = lam * y_a + (1 - lam) * y_b

        return mixed_x, mixed_y

    def cutmix(self, x, y):
        """Applies CutMix augmentation."""
        # Sample lambda from Beta distribution
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        # Generate bounding box
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)

        # Adjust lambda to match actual pixel ratio of the cut
        total_pixels = x.size(2) * x.size(3)
        cut_pixels = (bbx2 - bbx1) * (bby2 - bby1)
        lam = 1 - (cut_pixels / total_pixels)

        # Apply CutMix to images
        # Clone to avoid modifying the original tensor in place if it's shared
        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        # Mix labels
        y_a, y_b = y, y[index]
        mixed_y = lam * y_a + (1 - lam) * y_b

        return x_mixed, mixed_y

    def rand_bbox(self, size, lam):
        """Generates a random bounding box."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
