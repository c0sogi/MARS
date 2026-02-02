import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def load_and_process_data(csv_path, cache_name, load_cached_data=True):
    """
    Loads metadata CSV, processes labels into multi-hot vectors, and caches the result.
    Strictly follows the caching logic requirement.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, proceed to process from scratch
            pass

    # 2. Process data from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Process labels if they exist and we are not in test mode (test labels are placeholders)
    # We check if 'labels' column exists. For 'test', we usually don't need targets,
    # but if we wanted to process them, we could. Here we focus on train/val.
    if "labels" in df.columns:
        # We need to ensure we don't process the placeholder 'healthy' in test set as a ground truth
        # unless it's actually the ground truth.
        # However, for the dataset class, having the columns present is useful.

        # Initialize class columns
        class_to_idx = {cls: i for i, cls in enumerate(Config.CLASSES)}
        label_matrix = np.zeros((len(df), Config.NUM_CLASSES), dtype=np.float32)

        # Populate matrix
        for idx, row in df.iterrows():
            if pd.isna(row["labels"]):
                continue
            lbls = str(row["labels"]).split()
            for l in lbls:
                if l in class_to_idx:
                    label_matrix[idx, class_to_idx[l]] = 1.0

        # Assign columns to DataFrame
        for cls, col_idx in class_to_idx.items():
            df[cls] = label_matrix[:, col_idx]

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path)

    return df


def get_transforms(mode, cfg):
    """
    Returns Albumentations transforms for 'train' or 'val'/'test' modes.
    """
    if mode == "train":
        return A.Compose(
            [
                # RandomResizedCrop is a strong regularizer
                A.RandomResizedCrop(
                    height=cfg.IMG_SIZE, width=cfg.IMG_SIZE, scale=(0.08, 1.0)
                ),
                A.HorizontalFlip(p=0.5),
                # Color jitter for invariance to lighting conditions
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Resize to slightly larger then CenterCrop, or just Resize.
        # Standard ImageNet evaluation protocol: Resize(256) -> CenterCrop(224)
        return A.Compose(
            [
                A.Resize(height=256, width=256),
                A.CenterCrop(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class AppleDataset(Dataset):
    def __init__(self, df, transform=None, return_id=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (processed).
            transform (albumentations.Compose): Transforms to apply.
            return_id (bool): If True, returns (image, image_id). Used for test/submission.
                              If False, returns (image, target). Used for train/val.
        """
        self.df = df
        self.transform = transform
        self.return_id = return_id

        # Pre-construct full file paths
        self.file_paths = [
            os.path.join(Config.INPUT_ROOT, fp) for fp in df["file_path"].values
        ]
        self.image_ids = df["image"].values

        # Prepare targets if class columns exist
        self.targets = None
        if all(c in df.columns for c in Config.CLASSES):
            self.targets = df[Config.CLASSES].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return logic based on mode
        if self.return_id:
            return image, self.image_ids[idx]
        else:
            # Return targets
            if self.targets is not None:
                target = self.targets[idx]
            else:
                # Fallback for dummy targets if needed
                target = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
            return image, torch.tensor(target)


class MixupCutmix:
    """
    Applies MixUp or CutMix to a batch of images and targets.
    Returns mixed images and soft targets.
    """

    def __init__(self, cfg):
        self.use_mixup = cfg.USE_MIXUP
        self.mixup_alpha = cfg.MIXUP_ALPHA
        self.cutmix_alpha = cfg.CUTMIX_ALPHA
        self.mix_prob = cfg.MIX_PROB

    def __call__(self, batch_x, batch_y):
        """
        Args:
            batch_x: (B, C, H, W)
            batch_y: (B, NumClasses)
        """
        # Decide whether to apply augmentation
        if np.random.rand() > self.mix_prob:
            return batch_x, batch_y

        # Decide between MixUp and CutMix (50/50 split if both enabled)
        # Assuming both are enabled via config logic, or we can check flags.
        # Here we assume the class is instantiated only if at least one is desired.
        use_cutmix = np.random.rand() > 0.5

        if use_cutmix:
            return self._cutmix(batch_x, batch_y)
        else:
            return self._mixup(batch_x, batch_y)

    def _mixup(self, x, y):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        # Mix targets
        mixed_y = lam * y + (1 - lam) * y[index, :]
        return mixed_x, mixed_y

    def _cutmix(self, x, y):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        H, W = x.shape[2], x.shape[3]

        # Calculate bounding box
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Paste
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda to exact pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        # Mix targets
        mixed_y = lam * y + (1 - lam) * y[index, :]

        return x, mixed_y
