import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.tokenizer import Tokenizer


def get_transforms(phase: str):
    """
    Returns the image transformations for the specified phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Albumentations transformation pipeline.
    """
    # Standard ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                # Chemical structures are sensitive to flips/rotations (chirality),
                # so we avoid geometric augmentations that alter meaning.
                # We can use mild pixel-level augmentations.
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                        A.MotionBlur(blur_limit=3, p=0.5),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=0.5
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=0.5,
                        ),
                    ],
                    p=0.3,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Deterministic resizing and normalization
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class InChiDataset(Dataset):
    """
    PyTorch Dataset for loading chemical images and InChI labels.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: Tokenizer, transform=None):
        """
        Initialize the dataset.

        Args:
            df (pd.DataFrame): Dataframe containing 'image_id', 'InChI', and 'file_path'.
            tokenizer (Tokenizer): Instance of the Tokenizer class.
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.file_paths = df["file_path"].values
        self.labels = df["InChI"].values
        self.image_ids = df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load as BGR
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing files (though verification passed)
            # Create a black image to prevent crash
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback transform if none provided
            base_transform = A.Compose(
                [
                    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = base_transform(image=image)["image"]

        # 3. Process Label
        text = self.labels[idx]
        # Convert text to sequence of integers with padding
        # We use Config.MAX_LEN to ensure consistent tensor sizes
        sequence = self.tokenizer.text_to_sequence(
            text, max_len=Config.MAX_LEN, padding=True
        )

        # Calculate actual length (excluding padding) for masking purposes later
        # The sequence contains <SOS> ... tokens ... <EOS> <PAD> ...
        # Length is index of <EOS> + 1, or just count non-pad tokens
        seq_tensor = torch.tensor(sequence, dtype=torch.long)

        # Calculate valid length: count tokens that are not PAD
        # Note: text_to_sequence adds SOS and EOS.
        valid_len = (seq_tensor != self.tokenizer.pad_token_id).sum().item()

        return {
            "image": image,
            "seq": seq_tensor,
            "seq_len": torch.tensor(valid_len, dtype=torch.long),
            "image_id": self.image_ids[idx],
            "original_text": text,
        }


def collate_fn(batch):
    """
    Custom collate function to batch data.
    Since we pad in __getitem__, this mostly stacks tensors.
    """
    images = torch.stack([item["image"] for item in batch])
    seqs = torch.stack([item["seq"] for item in batch])
    seq_lens = torch.stack([item["seq_len"] for item in batch])
    image_ids = [item["image_id"] for item in batch]
    original_texts = [item["original_text"] for item in batch]

    return {
        "image": images,
        "seq": seqs,
        "seq_len": seq_lens,
        "image_id": image_ids,
        "original_text": original_texts,
    }
