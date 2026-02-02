import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from library.config import CFG

# ImageNet Mean and Std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class CassavaDataset(Dataset):
    """
    Dataset class for Cassava Leaf Disease Classification.
    Uses PIL for image loading as per strategy.
    """

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Metadata contains relative paths (e.g., 'train_images/xyz.jpg')
        # Input dir is './input'
        file_path = os.path.join(CFG.input_dir, row["file_path"])

        try:
            # Load image with PIL and convert to RGB
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt images
            print(f"Warning: Could not load image {file_path}. Error: {e}")
            # Return a black image of standard size
            image = Image.new("RGB", (600, 800), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # Return label if available, else dummy (for test set)
        if "label" in row:
            label = torch.tensor(row["label"], dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)

        return image, label


def get_transforms(phase, img_size):
    """
    Generates the transformation pipeline for a specific phase and image size.

    Args:
        phase (str): 'train', 'val', or 'test'.
        img_size (int): The target resolution (e.g., 224 or 384).
    """
    if phase == "train":
        return T.Compose(
            [
                # Geometric: Scale invariance
                T.RandomResizedCrop(img_size),
                # Orientation invariance
                T.RandomHorizontalFlip(),
                # Photometric diversity
                T.RandAugment(num_ops=2, magnitude=9),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        # Deterministic resizing for Validation/Test
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )


class Mixup:
    """
    Applies Mixup and CutMix to a batch of images and labels.
    Generates soft targets.
    """

    def __init__(
        self,
        mixup_p=0.5,
        cutmix_p=0.5,
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        num_classes=5,
    ):
        self.mixup_p = mixup_p
        self.cutmix_p = cutmix_p
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes = num_classes

    def __call__(self, batch_x, batch_y):
        """
        Args:
            batch_x: Input images tensor (B, C, H, W)
            batch_y: Input labels tensor (B) (integers)

        Returns:
            mixed_x: Tensor (B, C, H, W)
            mixed_y: Tensor (B, num_classes) (One-hot/Soft)
        """
        # Convert integer labels to one-hot
        batch_y_onehot = torch.zeros(
            batch_y.size(0), self.num_classes, device=batch_x.device
        )
        batch_y_onehot.scatter_(1, batch_y.view(-1, 1), 1)

        # Determine if we apply mixing
        # Strategy: 50% chance to apply mixing, 50% chance to keep clean
        # This balances regularization with the need for clean data
        if np.random.rand() > 0.5:
            return batch_x, batch_y_onehot

        # If mixing is selected, choose between MixUp and CutMix (50/50 split)
        use_cutmix = np.random.rand() > 0.5

        if use_cutmix:
            return self._cutmix(batch_x, batch_y_onehot)
        else:
            return self._mixup(batch_x, batch_y_onehot)

    def _mixup(self, x, y):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]
        return mixed_x, mixed_y

    def _cutmix(self, x, y):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        H, W = x.shape[2], x.shape[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Center of the box
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda to match exact pixel area removed
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        mixed_y = lam * y + (1 - lam) * y[index, :]
        return mixed_x, mixed_y


def load_metadata():
    """
    Loads train, validation, and test metadata dataframes.
    Handles debug sampling if configured.
    """
    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.val_csv)
    test_df = pd.read_csv(CFG.test_csv)

    if CFG.debug:
        print(f"Debug mode: Sampling {CFG.debug_sample_size} rows.")
        train_df = train_df.sample(
            n=min(len(train_df), CFG.debug_sample_size), random_state=CFG.seed
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), CFG.debug_sample_size), random_state=CFG.seed
        ).reset_index(drop=True)

    return train_df, val_df, test_df


def create_loaders(train_df, val_df, img_size, batch_size):
    """
    Creates DataLoaders for training and validation.
    """
    train_ds = CassavaDataset(train_df, transform=get_transforms("train", img_size))
    val_ds = CassavaDataset(val_df, transform=get_transforms("val", img_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
