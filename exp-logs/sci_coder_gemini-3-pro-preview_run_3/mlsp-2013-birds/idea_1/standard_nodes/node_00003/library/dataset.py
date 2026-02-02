import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from library import config


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification using Spectrograms.
    """

    def __init__(self, df, transform=None, phase="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transform (callable, optional): Optional transform to be applied on a sample.
            phase (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.phase = phase
        self.num_classes = config.NUM_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = int(row["rec_id"])

        # Construct spectrogram path
        # Original file_path: essential_data/src_wavs/filename.wav
        # Target path: supplemental_data/spectrograms/filename.bmp
        wav_path = row["file_path"]
        filename = os.path.basename(wav_path)
        bmp_filename = filename.replace(".wav", ".bmp")
        img_path = os.path.join(config.SPECTROGRAM_DIR, bmp_filename)

        # Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Fallback for missing files (should not happen based on metadata check)
            image = Image.new("RGB", config.IMG_SIZE)

        if self.transform:
            image = self.transform(image)

        # Extract Labels
        if self.phase in ["train", "val"]:
            label_str = str(row["labels"])
            label_vec = np.zeros(self.num_classes, dtype=np.float32)

            if label_str != "?" and label_str.lower() != "nan":
                try:
                    indices = [int(x) for x in label_str.split()]
                    for cls_idx in indices:
                        if 0 <= cls_idx < self.num_classes:
                            label_vec[cls_idx] = 1.0
                except ValueError:
                    pass

            labels_tensor = torch.tensor(label_vec, dtype=torch.float32)
            return image, labels_tensor

        else:
            return image, torch.tensor(rec_id, dtype=torch.long)


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for train, val, and test sets.
    """
    set_seed(config.RANDOM_SEED)

    # 1. Load Metadata
    train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(config.TEST_METADATA_PATH)

    # 2. Define Transforms
    # Basic transforms for now; can add augmentation later
    common_transforms = transforms.Compose(
        [
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_transforms = transforms.Compose(
        [
            transforms.Resize(config.IMG_SIZE),
            # Add some augmentation for training
            transforms.RandomHorizontalFlip(
                p=0.5
            ),  # Time reversal might be valid for some bird calls?
            # Actually, bird calls are temporal, so flipping might change meaning.
            # But for detection it might help regularization.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 3. Create Datasets
    train_dataset = BirdDataset(train_meta, transform=train_transforms, phase="train")
    val_dataset = BirdDataset(val_meta, transform=common_transforms, phase="val")
    test_dataset = BirdDataset(test_meta, transform=common_transforms, phase="test")

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
