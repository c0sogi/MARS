import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


class CassavaDataset(Dataset):
    """
    Custom Dataset for Cassava Leaf Disease Classification.
    Reads images using OpenCV and applies Albumentations transforms.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        input_root: str,
        transforms=None,
        mode: str = "train",
        output_label: bool = True,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path' and 'label'.
            input_root (str): Root directory where images are stored.
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
            output_label (bool): Whether to return the label.
        """
        self.df = df.reset_index(drop=True).copy()
        self.input_root = input_root
        self.transforms = transforms
        self.mode = mode
        self.output_label = output_label

        # Pre-check file existence to avoid runtime errors, or just trust metadata
        # Given the metadata generation script checks this, we assume paths are valid relative to input_root.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        # Construct full file path
        # metadata file_path is relative to input_root (e.g., "train_images/xyz.jpg")
        img_path = os.path.join(self.input_root, row["file_path"])

        # Load Image
        img = cv2.imread(img_path)
        if img is None:
            # Fallback or error handling; for this task we assume data integrity
            raise FileNotFoundError(f"Image not found at {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            img = self.transforms(image=img)["image"]
        else:
            # Minimal transform if none provided
            T = A.Compose([A.Normalize(), ToTensorV2()])
            img = T(image=img)["image"]

        # Return Logic
        if self.output_label:
            label = row["label"]
            return img, torch.tensor(label, dtype=torch.long)
        else:
            return img


class Mixup:
    """
    Implements Mixup and CutMix augmentation.
    """

    def __init__(
        self,
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 0.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        mode: str = "batch",
        num_classes: int = 5,
    ):
        """
        Args:
            mixup_alpha (float): Mixup alpha value.
            cutmix_alpha (float): Cutmix alpha value.
            prob (float): Probability of applying mixup or cutmix.
            switch_prob (float): Probability of switching to cutmix instead of mixup.
            mode (str): 'batch' (apply same lambda to whole batch) or 'elem' (per element).
            num_classes (int): Number of classes for one-hot encoding.
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
        self.num_classes = num_classes

    def __call__(self, x, target):
        """
        Args:
            x (torch.Tensor): Input batch of images (N, C, H, W).
            target (torch.Tensor): Input batch of labels (N).

        Returns:
            x (torch.Tensor): Mixed images.
            target (torch.Tensor): Mixed soft labels (N, num_classes).
        """
        # Convert target to one-hot
        target = torch.eye(self.num_classes, device=x.device)[target]

        if np.random.rand() > self.prob:
            return x, target

        # Determine whether to use Mixup or Cutmix
        use_cutmix = np.random.rand() < self.switch_prob

        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        if alpha <= 0:
            return x, target

        lam = np.random.beta(alpha, alpha)

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        if use_cutmix:
            # CutMix
            # Generate bounding box
            cx = np.random.uniform(0, x.shape[3])
            cy = np.random.uniform(0, x.shape[2])
            w = x.shape[3] * np.sqrt(1 - lam)
            h = x.shape[2] * np.sqrt(1 - lam)
            x0 = int(np.round(max(cx - w / 2, 0)))
            x1 = int(np.round(min(cx + w / 2, x.shape[3])))
            y0 = int(np.round(max(cy - h / 2, 0)))
            y1 = int(np.round(min(cy + h / 2, x.shape[2])))

            # Adjust lambda to exact area ratio
            lam = 1 - ((x1 - x0) * (y1 - y0) / (x.shape[3] * x.shape[2]))

            x[:, :, y0:y1, x0:x1] = x[index, :, y0:y1, x0:x1]
        else:
            # MixUp
            x = lam * x + (1 - lam) * x[index]

        # Mix targets
        target = lam * target + (1 - lam) * target[index]

        return x, target


def get_transforms(img_size: int, data: str = "train"):
    """
    Returns Albumentations transforms based on the data phase.

    Args:
        img_size (int): Target image size (height and width).
        data (str): 'train' or 'valid'/'test'.
    """
    if data == "train":
        return A.Compose(
            [
                # Contextual Cropping: Scale 0.3 to 1.0
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.3, 1.0), p=1.0),
                # Geometric Augmentations
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                # Normalization and Tensor conversion
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    else:
        # Fallback
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )


def get_loaders(config: Config, phase: str = "coarse"):
    """
    Creates DataLoaders for training and validation.

    Args:
        config (Config): Configuration object.
        phase (str): 'coarse' or 'fine' to determine image size and batch size.

    Returns:
        train_loader, val_loader, mixup_fn
    """
    # Determine parameters based on phase
    if phase == "coarse":
        img_size = config.img_size_coarse
        batch_size = config.batch_size_coarse
    elif phase == "fine":
        img_size = config.img_size_fine
        batch_size = config.batch_size_fine
    else:
        # Default fallback (e.g. for inference if not specified)
        img_size = config.img_size_fine
        batch_size = config.batch_size_fine

    # Load Metadata
    df_train = pd.read_csv(config.train_metadata)
    df_val = pd.read_csv(config.val_metadata)

    # Debugging: Subset data if debug mode is on
    if config.debug:
        df_train = df_train.head(config.debug_subset_size)
        df_val = df_val.head(config.debug_subset_size)

    # Create Datasets
    train_dataset = CassavaDataset(
        df=df_train,
        input_root=config.input_root,
        transforms=get_transforms(img_size, data="train"),
        mode="train",
        output_label=True,
    )

    val_dataset = CassavaDataset(
        df=df_val,
        input_root=config.input_root,
        transforms=get_transforms(img_size, data="valid"),
        mode="val",
        output_label=True,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,  # Important for Batch Norm stability with small batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Initialize Mixup
    mixup_fn = None
    if config.mixup_prob > 0:
        mixup_fn = Mixup(
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
            prob=config.mixup_prob,
            switch_prob=config.mixup_switch_prob,
            mode="batch",
            num_classes=config.num_classes,
        )

    return train_loader, val_loader, mixup_fn


def get_test_loader(config: Config, phase: str = "fine"):
    """
    Creates DataLoader for testing/inference.

    Args:
        config (Config): Configuration object.
        phase (str): 'coarse' or 'fine' to determine image size.

    Returns:
        test_loader
    """
    # Use fine resolution for best inference results usually
    if phase == "coarse":
        img_size = config.img_size_coarse
        batch_size = config.batch_size_coarse
    else:
        img_size = config.img_size_fine
        batch_size = config.batch_size_fine

    df_test = pd.read_csv(config.test_metadata)

    test_dataset = CassavaDataset(
        df=df_test,
        input_root=config.input_root,
        transforms=get_transforms(img_size, data="test"),
        mode="test",
        output_label=False,  # Test set usually has dummy labels
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
