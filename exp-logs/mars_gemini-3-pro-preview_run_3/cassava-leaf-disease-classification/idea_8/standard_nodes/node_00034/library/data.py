import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.
    size: (Batch, Channel, Height, Width)
    lam: lambda value from Beta distribution
    """
    H = size[2]
    W = size[3]
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


class CassavaDataset(Dataset):
    def __init__(self, df, root_dir, transforms=None, output_label=True):
        self.df = df
        self.root_dir = root_dir
        self.transforms = transforms
        self.output_label = output_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # file_path is relative, e.g., "train_images/1000015157.jpg"
        img_path = os.path.join(self.root_dir, row["file_path"])

        img = cv2.imread(img_path)
        if img is None:
            # Fallback for missing images to maintain stability
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        if self.output_label:
            label = row["label"]
            return img, torch.tensor(label, dtype=torch.long)
        else:
            # Return image_id for test set tracking
            return img, row["image_id"]


def get_transforms(data_type, cfg):
    if data_type == "train":
        aug_list = [
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        ]

        if cfg.USE_GEOMETRIC_AUG:
            aug_list.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Transpose(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5
                    ),
                ]
            )

        aug_list.extend(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
        return A.Compose(aug_list)

    elif data_type == "valid" or data_type == "test":
        return A.Compose(
            [
                A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class MixupCutmixCollator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_classes = cfg.NUM_CLASSES
        self.use_mixing = cfg.USE_MIXING
        self.mix_prob = cfg.MIX_PROB
        self.mixup_alpha = cfg.MIXUP_ALPHA
        self.cutmix_alpha = cfg.CUTMIX_ALPHA

    def __call__(self, batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs)
        labels = torch.stack(labels)

        # If mixing is disabled or random roll fails, return standard batch
        if not self.use_mixing or np.random.rand() > self.mix_prob:
            return imgs, labels

        batch_size = imgs.size(0)
        indices = torch.randperm(batch_size)

        shuffled_imgs = imgs[indices]
        shuffled_labels = labels[indices]

        # Create one-hot targets for the mix
        target_a = torch.zeros(
            batch_size, self.num_classes, device=imgs.device
        ).scatter_(1, labels.view(-1, 1), 1.0)
        target_b = torch.zeros(
            batch_size, self.num_classes, device=imgs.device
        ).scatter_(1, shuffled_labels.view(-1, 1), 1.0)

        # Decide MixUp vs CutMix (50/50)
        choice = np.random.rand()

        if choice < 0.5:
            # MixUp
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            imgs = imgs * lam + shuffled_imgs * (1 - lam)
            targets = target_a * lam + target_b * (1 - lam)
        else:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.shape, lam)

            # Adjust lambda to exact area ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (imgs.shape[-1] * imgs.shape[-2])
            )

            imgs[:, :, bbx1:bbx2, bby1:bby2] = shuffled_imgs[:, :, bbx1:bbx2, bby1:bby2]
            targets = target_a * lam + target_b * (1 - lam)

        return imgs, targets


def get_dataloaders(cfg):
    # Load Metadata
    train_df = pd.read_csv(cfg.TRAIN_METADATA)
    val_df = pd.read_csv(cfg.VAL_METADATA)
    test_df = pd.read_csv(cfg.TEST_METADATA)

    if cfg.DEBUG:
        train_df = train_df.iloc[: cfg.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: cfg.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: cfg.DEBUG_SAMPLE_SIZE]

    # Datasets
    train_dataset = CassavaDataset(
        train_df,
        cfg.INPUT_ROOT,
        transforms=get_transforms("train", cfg),
        output_label=True,
    )

    val_dataset = CassavaDataset(
        val_df,
        cfg.INPUT_ROOT,
        transforms=get_transforms("valid", cfg),
        output_label=True,
    )

    test_dataset = CassavaDataset(
        test_df,
        cfg.INPUT_ROOT,
        transforms=get_transforms("test", cfg),
        output_label=False,
    )

    # Collator for training
    train_collator = MixupCutmixCollator(cfg)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
