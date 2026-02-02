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
    Cite solution_lesson_node_00003: Treating audio as images using spectrograms.
    """

    def __init__(self, df, phase="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            phase (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.num_classes = config.NUM_CLASSES

        # Standard ImageNet normalization
        # Cite solution_lesson_node_00006: Avoiding temporal distortion (no flips).
        if phase == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomAffine(degrees=0, translate=(0.2, 0)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = int(row["rec_id"])

        # Construct spectrogram path
        # Original file_path: essential_data/src_wavs/filename.wav
        # Spectrogram path: supplemental_data/spectrograms/filename.bmp
        wav_path = row["file_path"]
        filename = os.path.basename(wav_path).replace(".wav", ".bmp")
        img_path = os.path.join(config.SPECTROGRAM_DIR, filename)

        # Load Image
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image)
        except Exception as e:
            # Fallback for missing files
            print(f"Warning: Could not load {img_path}: {e}")
            image_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

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
            return image_tensor, labels_tensor

        else:
            return image_tensor, torch.tensor(rec_id, dtype=torch.long)


def get_dataloaders():
    """
    Prepares DataLoaders for train, val, and test sets.
    """
    set_seed(config.RANDOM_SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # 2. Create Datasets (Cite solution_lesson_node_00001: Avoiding global feature aggregation)
    train_dataset = BirdDataset(train_df, phase="train")
    val_dataset = BirdDataset(val_df, phase="val")
    test_dataset = BirdDataset(test_df, phase="test")

    # 3. Create DataLoaders
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
