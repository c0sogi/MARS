import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Loads images from disk and converts labels to multi-hot vectors.
    """

    def __init__(self, metadata_path, transform=None, mode="train"):
        self.df = pd.read_csv(metadata_path)
        self.transform = transform
        self.mode = mode
        self.classes = Config.CLASSES
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Debugging: Use a small subset if enabled
        if Config.DEBUG:
            self.df = self.df.head(Config.DEBUG_SUBSET_SIZE).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct full path: input_dir + relative_path_from_metadata
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback transform (Resize + Normalize + ToTensor)
            fallback = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
            image = fallback(image=image)["image"]

        # Process labels
        # Initialize zero vector
        label_vector = np.zeros(self.num_classes, dtype=np.float32)

        # In test mode, labels might be placeholders, but we parse them if they exist
        # In train/val, we parse the space-delimited string
        if "labels" in row and isinstance(row["labels"], str):
            labels = row["labels"].split()
            for l in labels:
                if l in self.class_to_idx:
                    label_vector[self.class_to_idx[l]] = 1.0

        return image, torch.tensor(label_vector)


class MixupCutmixCollate:
    """
    Custom collate function to apply Mixup and CutMix augmentation on batches.
    Generates soft targets for training.
    """

    def __init__(self):
        self.mixup_alpha = Config.MIXUP_ALPHA
        self.cutmix_alpha = Config.CUTMIX_ALPHA
        self.prob = Config.MIXUP_PROB
        self.enabled = Config.MIXUP_ENABLED

    def rand_bbox(self, size, lam):
        """Generates a random bounding box for CutMix."""
        H = size[2]
        W = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniformly sample center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch):
        images, targets = list(zip(*batch))
        images = torch.stack(images)
        targets = torch.stack(targets)

        # Skip if disabled or probability check fails
        if not self.enabled or np.random.rand() > self.prob:
            return images, targets

        batch_size = images.size(0)
        indices = torch.randperm(batch_size)

        # Randomly choose between Mixup and CutMix
        use_cutmix = np.random.rand() > 0.5

        if use_cutmix:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)

            # Adjust lambda to match the exact pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2))
            )

            # Apply patch replacement
            images[:, :, bby1:bby2, bbx1:bbx2] = images[
                indices, :, bby1:bby2, bbx1:bbx2
            ]
            # Mix targets
            targets = lam * targets + (1 - lam) * targets[indices]

        else:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            # Mix images
            images = lam * images + (1 - lam) * images[indices]
            # Mix targets
            targets = lam * targets + (1 - lam) * targets[indices]

        return images, targets


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the specified mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Robust augmentations for larger model capacity (Cite solution_lesson_node_00002)
                A.RandomRotate90(p=0.5),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=4,
                    min_height=16,
                    min_width=16,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_loaders():
    """
    Creates and returns Training and Validation DataLoaders.
    """
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    train_dataset = AppleDataset(
        Config.TRAIN_METADATA, transform=train_transform, mode="train"
    )
    val_dataset = AppleDataset(Config.VAL_METADATA, transform=val_transform, mode="val")

    # Use custom collate function for training to enable Mixup/CutMix
    train_collate = MixupCutmixCollate()

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=train_collate,
        drop_last=True,  # Important for BatchNorm stability with Mixup
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader():
    """
    Creates and returns the Test DataLoader.
    """
    test_transform = get_transforms("test")
    test_dataset = AppleDataset(
        Config.TEST_METADATA, transform=test_transform, mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
