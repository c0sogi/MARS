import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(data_type="train"):
    """
    Returns the albumentations transform pipeline based on the data type.

    Args:
        data_type (str): 'train' for heavy augmentation, 'val' or 'test' for standard resizing/normalization.

    Returns:
        albumentations.Compose: The transform pipeline.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class CassavaDataset(Dataset):
    """
    Dataset class for Cassava Leaf Disease Classification.
    Reads images via OpenCV and applies Albumentations transforms.
    """

    def __init__(self, df, transforms=None, output_label=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, label, file_path).
            transforms (albumentations.Compose): Transforms to apply.
            output_label (bool): Whether to return the label.
        """
        self.df = df
        self.transforms = transforms
        self.output_label = output_label

        # Pre-compute full paths to avoid overhead in __getitem__
        # file_path in metadata is relative to input root (e.g., "train_images/xyz.jpg")
        self.file_paths = [
            os.path.join(Config.INPUT_ROOT, fp) for fp in df["file_path"].values
        ]

        if self.output_label:
            self.labels = df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load Image
        img_path = self.file_paths[index]
        img = cv2.imread(img_path)

        if img is None:
            # Fallback for missing/corrupt images to ensure robustness
            # Create a black image of expected size
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]

        # Return
        if self.output_label:
            label = self.labels[index]
            return img, torch.tensor(label, dtype=torch.long)
        else:
            return img


class Mixup:
    """
    Implements Mixup and CutMix regularization.
    Designed to be called on a batch of (images, targets) within the training loop.
    """

    def __init__(
        self,
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        prob=Config.MIXUP_PROB,
        num_classes=Config.NUM_CLASSES,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes

    def rand_bbox(self, size, lam):
        """
        Generates a random bounding box for CutMix.
        """
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniformly sample center of the box
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch_x, batch_y):
        """
        Applies Mixup or CutMix to the batch.

        Args:
            batch_x (torch.Tensor): Batch of images [B, C, H, W]
            batch_y (torch.Tensor): Batch of labels [B]

        Returns:
            mixed_x (torch.Tensor): Augmented images
            mixed_y (torch.Tensor): Soft labels [B, Num_Classes]
        """
        # Convert integer targets to one-hot float tensors
        batch_y = torch.nn.functional.one_hot(
            batch_y, num_classes=self.num_classes
        ).float()

        # Skip mixing based on probability
        if np.random.rand() > self.prob:
            return batch_x, batch_y

        batch_size = batch_x.size(0)
        rand_index = torch.randperm(batch_size).to(batch_x.device)

        # Decide between Mixup and CutMix (50/50 split when mixing is active)
        use_cutmix = np.random.rand() < 0.5

        if use_cutmix:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(batch_x.size(), lam)

            # Adjust lambda to match the exact pixel ratio of the cut
            lam = 1 - (
                (bbx2 - bbx1)
                * (bby2 - bby1)
                / (batch_x.size()[-1] * batch_x.size()[-2])
            )

            mixed_x = batch_x.clone()
            mixed_x[:, :, bbx1:bbx2, bby1:bby2] = batch_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # Mix labels
            mixed_y = lam * batch_y + (1 - lam) * batch_y[rand_index]
        else:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixed_x = lam * batch_x + (1 - lam) * batch_x[rand_index]
            mixed_y = lam * batch_y + (1 - lam) * batch_y[rand_index]

        return mixed_x, mixed_y
