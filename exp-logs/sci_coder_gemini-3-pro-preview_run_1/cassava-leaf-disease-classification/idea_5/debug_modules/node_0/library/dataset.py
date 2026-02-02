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
    Loads images via OpenCV and applies Albumentations transforms.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transforms=None,
        output_label: bool = True,
        input_root: str = "./input",
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'label' (optional).
            transforms (albumentations.Compose): Albumentations transformations.
            output_label (bool): Whether to return the label (True for train/val, False for test).
            input_root (str): Root directory for input data.
        """
        self.df = df
        self.transforms = transforms
        self.output_label = output_label
        self.input_root = input_root

        # Pre-resolve full paths to avoid doing os.path.join in __getitem__
        # The metadata 'file_path' is relative to input_root
        self.file_paths = [
            os.path.join(input_root, fp) for fp in df["file_path"].values
        ]

        if self.output_label:
            self.labels = df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        # Load Image
        path = self.file_paths[index]
        img = cv2.imread(path)

        if img is None:
            # Fallback for missing/corrupt images, though metadata check should prevent this
            # Return a black image or raise error. Here we raise error to fail fast.
            raise FileNotFoundError(f"Image not found at {path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]

        # Return Data
        if self.output_label:
            label = self.labels[index]
            return img, torch.tensor(label, dtype=torch.long)
        else:
            return img


def get_transforms(data_split: str, size: int, config: Config):
    """
    Generates the Albumentations transform pipeline.

    Args:
        data_split (str): 'train', 'valid', or 'test'.
        size (int): Target image size (height and width).
        config (Config): Configuration object.

    Returns:
        A.Compose: The transform pipeline.
    """
    if data_split == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    height=size,
                    width=size,
                    scale=config.random_resized_crop_scale,
                    p=1.0,
                ),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data_split == "valid" or data_split == "test":
        return A.Compose(
            [
                A.Resize(height=size, width=size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data_split: {data_split}")


class Mixup:
    """
    Implements Mixup and Cutmix augmentation.
    """

    def __init__(self, config: Config):
        self.mixup_alpha = config.mixup_alpha
        self.cutmix_alpha = config.cutmix_alpha
        self.mixup_prob = config.mixup_prob
        self.switch_prob = config.mixup_switch_prob
        self.num_classes = config.num_classes
        self.device = config.device

    def one_hot(self, y, num_classes, dtype=torch.float32):
        # Create one-hot encoding
        # y is shape (B,)
        # output is shape (B, C)
        return torch.zeros(
            y.size(0), num_classes, device=y.device, dtype=dtype
        ).scatter_(1, y.view(-1, 1), 1)

    def rand_bbox(self, img_shape, lam):
        """Generate random bounding box for CutMix."""
        W = img_shape[2]
        H = img_shape[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, x, y):
        """
        Args:
            x (torch.Tensor): Input batch of images (B, C, H, W).
            y (torch.Tensor): Input batch of labels (B,).

        Returns:
            x (torch.Tensor): Mixed images.
            y (torch.Tensor): Mixed labels (soft targets, B, C).
        """
        # Convert targets to one-hot
        y = self.one_hot(y, self.num_classes)

        # Decide whether to apply mixup/cutmix based on probability
        if np.random.rand() > self.mixup_prob:
            return x, y

        # Decide between Mixup and Cutmix
        use_cutmix = np.random.rand() < self.switch_prob

        # Get lambda
        if use_cutmix:
            alpha = self.cutmix_alpha
        else:
            alpha = self.mixup_alpha

        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        if use_cutmix:
            # CutMix
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

            # Adjust lambda to match actual pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

            # Mix targets
            y = lam * y + (1 - lam) * y[index]
        else:
            # MixUp
            x = lam * x + (1 - lam) * x[index]
            y = lam * y + (1 - lam) * y[index]

        return x, y


def get_dataloaders(config: Config, stage: int = 1):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        config (Config): Configuration object.
        stage (int): 1 or 2, determines the image resolution.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Determine resolution based on stage
    if stage == 1:
        img_size = config.input_size_stage1
    elif stage == 2:
        img_size = config.input_size_stage2
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")

    # Load Metadata
    df_train = pd.read_csv(config.train_metadata_path)
    df_val = pd.read_csv(config.val_metadata_path)
    df_test = pd.read_csv(config.test_metadata_path)

    # Debug Sampling
    if config.debug:
        df_train = df_train.sample(
            frac=config.data_subset_fraction, random_state=config.seed
        ).reset_index(drop=True)
        df_val = df_val.sample(
            frac=config.data_subset_fraction, random_state=config.seed
        ).reset_index(drop=True)
        # We usually keep test set intact or sample it too, but for submission structure validity,
        # usually we want to run on full test if possible, or sample if strictly debugging pipeline.
        # Here we sample to speed up debug cycle.
        df_test = df_test.sample(
            frac=config.data_subset_fraction, random_state=config.seed
        ).reset_index(drop=True)

        print(
            f"DEBUG MODE: Sampled datasets. Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
        )

    # Define Transforms
    train_transforms = get_transforms("train", img_size, config)
    val_transforms = get_transforms("valid", img_size, config)
    test_transforms = get_transforms("test", img_size, config)

    # Create Datasets
    train_dataset = CassavaDataset(
        df_train,
        transforms=train_transforms,
        output_label=True,
        input_root=config.input_root,
    )

    val_dataset = CassavaDataset(
        df_val,
        transforms=val_transforms,
        output_label=True,
        input_root=config.input_root,
    )

    test_dataset = CassavaDataset(
        df_test,
        transforms=test_transforms,
        output_label=False,
        input_root=config.input_root,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,  # Important for Batch Norm stability and Mixup
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
