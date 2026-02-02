import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG
from library.utils import seed_worker


def load_metadata(mode="train", load_cached_data=True):
    """
    Loads metadata for the specified mode (train/val/test).
    Implements caching mechanism using Parquet files.
    """
    # Ensure output directory exists
    os.makedirs(CFG.output_dir, exist_ok=True)

    cache_path = os.path.join(CFG.output_dir, f"{mode}_meta.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Reloading from source.")

    # 2. Load from source
    if mode == "train":
        csv_path = CFG.train_csv
    elif mode == "val":
        csv_path = CFG.val_csv
    elif mode == "test":
        csv_path = CFG.test_csv
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Failed to save cache for {mode}: {e}")

    return df


def get_transforms(phase, img_size):
    """
    Returns Albumentations transforms for the specified phase and image size.
    """
    if phase == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.3, 1.0), p=1.0),
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
    else:
        # Validation and Test
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


class CassavaDataset(Dataset):
    def __init__(self, df, transform=None, output_label=True):
        self.df = df
        self.transform = transform
        self.output_label = output_label

        # Pre-calculate full paths
        self.file_paths = df["file_path"].values

        # Handle labels if required
        if output_label:
            self.labels = df["label"].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Resolve full path relative to input root
        rel_path = self.file_paths[idx]
        file_path = os.path.join(CFG.input_root, rel_path)

        # Read image
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {file_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return result
        if self.output_label:
            label = self.labels[idx]
            return image, label
        else:
            return image


class MixupCollate:
    """
    Collate function that applies Mixup and Cutmix regularization.
    Returns images and soft targets.
    """

    def __init__(
        self,
        num_classes,
        mixup_alpha=0.4,
        cutmix_alpha=1.0,
        prob=0.5,
        label_smoothing=0.0,
    ):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.label_smoothing = label_smoothing

    def __call__(self, batch):
        images, targets = list(zip(*batch))
        images = torch.stack(images)
        targets = torch.tensor(targets, device=images.device)

        batch_size = images.size(0)

        # Prepare one-hot targets with label smoothing
        with torch.no_grad():
            true_dist = torch.zeros(batch_size, self.num_classes, device=images.device)
            true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            targets = true_dist

        # Decide whether to apply mixup/cutmix
        if np.random.rand() > self.prob:
            return images, targets

        # Decide between Mixup and Cutmix
        use_cutmix = np.random.rand() > 0.5

        # Generate permutation
        rand_index = torch.randperm(batch_size)
        target_a = targets
        target_b = targets[rand_index]

        if use_cutmix:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            # Ensure lambda is within range to avoid empty boxes
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)

            # Adjust lambda to exact area ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2])
            )

            images[:, :, bbx1:bbx2, bby1:bby2] = images[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]
        else:
            # MixUp
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            images = lam * images + (1 - lam) * images[rand_index]

        # Mix targets
        targets = lam * target_a + (1 - lam) * target_b

        return images, targets

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


def get_loaders(img_size, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Load DataFrames
    train_df = load_metadata("train", load_cached_data)
    val_df = load_metadata("val", load_cached_data)
    test_df = load_metadata("test", load_cached_data)

    # Define Transforms
    train_transforms = get_transforms("train", img_size)
    val_transforms = get_transforms("val", img_size)

    # Create Datasets
    train_dataset = CassavaDataset(
        train_df, transform=train_transforms, output_label=True
    )
    val_dataset = CassavaDataset(val_df, transform=val_transforms, output_label=True)
    test_dataset = CassavaDataset(
        test_df, transform=val_transforms, output_label=False
    )  # No labels needed for inference usually, but we keep structure

    # Define Collate Function for Training
    mixup_collate = MixupCollate(
        num_classes=CFG.num_classes,
        mixup_alpha=CFG.mixup_alpha,
        cutmix_alpha=CFG.cutmix_alpha,
        prob=CFG.mixup_prob,
        label_smoothing=CFG.label_smoothing,
    )

    # Create DataLoaders
    # Generator for reproducibility
    g = torch.Generator()
    g.manual_seed(CFG.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=mixup_collate,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
