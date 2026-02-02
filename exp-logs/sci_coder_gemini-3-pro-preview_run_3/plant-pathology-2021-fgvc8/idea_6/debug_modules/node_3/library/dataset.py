import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, default_collate

try:
    from timm.data.mixup import rand_bbox
except ImportError:
    # Fallback if import fails
    def rand_bbox(img_shape, lam, margin=0.0, count=None):
        ratio = np.sqrt(1 - lam)
        img_h, img_w = img_shape[-2:]
        cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
        cx = np.random.randint(0, img_w)
        cy = np.random.randint(0, img_h)
        yl = np.clip(cy - cut_h // 2, 0, img_h)
        yh = np.clip(cy + cut_h // 2, 0, img_h)
        xl = np.clip(cx - cut_w // 2, 0, img_w)
        xh = np.clip(cx + cut_w // 2, 0, img_w)
        return yl, yh, xl, xh


from library.config import Config
from library.utils import seed_everything


def load_and_process_metadata(csv_path, cache_path, load_cached_data=True):
    """
    Loads metadata from CSV or Parquet cache.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}")

    # Process from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Save to cache
    try:
        df.to_parquet(cache_path)
    except Exception as e:
        print(f"Failed to save cache to {cache_path}: {e}")

    return df


def get_transforms(mode, cfg):
    """
    Returns Albumentations transforms for train, val, or test.
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
                    scale=cfg.RANDOM_RESIZE_CROP_SCALE,
                ),
                A.HorizontalFlip(p=cfg.FLIP_PROB),
                A.VerticalFlip(p=cfg.FLIP_PROB),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class AppleDataset(Dataset):
    def __init__(self, df, transforms, root_dir, label2id):
        self.df = df
        self.transforms = transforms
        self.root_dir = root_dir
        self.label2id = label2id
        self.num_classes = len(label2id)
        self.has_labels = "labels" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata 'file_path' is relative to input dir (e.g. "train_images/img.jpg")
        rel_path = row["file_path"]
        full_path = os.path.join(self.root_dir, rel_path)

        # Load Image
        image = cv2.imread(full_path)
        if image is None:
            # Handle missing image gracefully or raise error
            raise FileNotFoundError(f"Image not found at {full_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Process Labels
        if self.has_labels:
            labels_str = row["labels"]
            label_vec = np.zeros(self.num_classes, dtype=np.float32)

            # Split space-delimited labels
            # Example: "scab frog_eye_leaf_spot"
            current_labels = labels_str.split()
            for lbl in current_labels:
                if lbl in self.label2id:
                    label_vec[self.label2id[lbl]] = 1.0

            return image, torch.tensor(label_vec)
        else:
            # For test set where labels might be placeholders or ignored
            return image, torch.tensor([])


class MultiLabelMixup:
    """
    Custom Mixup/CutMix implementation for Multi-Label (pre-encoded) targets.
    timm's Mixup assumes targets are indices and re-encodes them, which breaks multi-label.
    """

    def __init__(self, cfg):
        self.mixup_alpha = cfg.MIXUP_ALPHA
        self.cutmix_alpha = cfg.CUTMIX_ALPHA
        self.prob = cfg.MIXUP_PROB
        self.switch_prob = cfg.SWITCH_PROB

    def __call__(self, x, target):
        # x: (B, C, H, W)
        # target: (B, NumClasses) - Float tensor

        if np.random.rand() > self.prob:
            return x, target

        # Determine mode (Mixup vs CutMix)
        use_cutmix = False
        if self.cutmix_alpha > 0 and self.mixup_alpha > 0:
            use_cutmix = np.random.rand() < self.switch_prob
        elif self.cutmix_alpha > 0:
            use_cutmix = True

        # Sample Lambda
        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        # Apply
        if use_cutmix:
            (yl, yh, xl, xh) = rand_bbox(x.shape, lam)
            # Adjust lambda to match exact pixel ratio
            lam = 1 - ((xh - xl) * (yh - yl) / (x.shape[-1] * x.shape[-2]))

            # Mix Images (CutMix)
            # We flip the batch to get the "other" image (standard timm practice for 'batch' mode)
            x_flipped = x.flip(0)
            x[:, :, yl:yh, xl:xh] = x_flipped[:, :, yl:yh, xl:xh]

            # Mix Targets
            target_flipped = target.flip(0)
            target = target * lam + target_flipped * (1.0 - lam)

        else:
            # Mixup
            x_flipped = x.flip(0)
            x = x * lam + x_flipped * (1.0 - lam)

            target_flipped = target.flip(0)
            target = target * lam + target_flipped * (1.0 - lam)

        return x, target


class MixupCollate:
    """
    Collate function that applies MixUp/CutMix to the batch.
    """

    def __init__(self, cfg):
        self.mixup = MultiLabelMixup(cfg)

    def __call__(self, batch):
        # Default collate stacks images and labels
        batch = default_collate(batch)
        images, labels = batch

        # Apply Mixup
        images, labels = self.mixup(images, labels)

        return images, labels


def get_train_val_loaders(cfg, load_cached_data=True):
    """
    Creates and returns the training and validation DataLoaders.
    """
    # Load Metadata
    train_df = load_and_process_metadata(
        cfg.TRAIN_METADATA_PATH, cfg.TRAIN_CACHE_PATH, load_cached_data
    )
    val_df = load_and_process_metadata(
        cfg.VAL_METADATA_PATH, cfg.VAL_CACHE_PATH, load_cached_data
    )

    # Transforms
    train_transforms = get_transforms("train", cfg)
    val_transforms = get_transforms("val", cfg)

    # Datasets
    train_dataset = AppleDataset(
        train_df, train_transforms, cfg.INPUT_DIR, cfg.LABEL2ID
    )
    val_dataset = AppleDataset(val_df, val_transforms, cfg.INPUT_DIR, cfg.LABEL2ID)

    # Collate Function (Only for Train if MixUp is enabled)
    collate_fn = MixupCollate(cfg) if cfg.MIXUP_PROB > 0 else None

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(cfg, load_cached_data=True):
    """
    Creates and returns the test DataLoader.
    """
    test_df = load_and_process_metadata(
        cfg.TEST_METADATA_PATH, cfg.TEST_CACHE_PATH, load_cached_data
    )
    test_transforms = get_transforms("test", cfg)

    test_dataset = AppleDataset(test_df, test_transforms, cfg.INPUT_DIR, cfg.LABEL2ID)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
