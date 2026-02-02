import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.tokenizer import Tokenizer


def get_transforms(phase: str):
    """
    Returns the image transformation pipeline for the specified phase.

    Args:
        phase (str): One of 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Albumentations composition of transforms.
    """
    # Standard normalization using ImageNet stats defined in Config
    # and resizing to the target input size.
    return A.Compose(
        [
            A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
            A.Normalize(
                mean=Config.NORM_MEAN,
                std=Config.NORM_STD,
            ),
            ToTensorV2(),
        ]
    )


class InChiDataset(Dataset):
    """
    PyTorch Dataset for the InChI prediction task.
    """

    def __init__(self, df, tokenizer, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata (image_id, file_path, InChI).
            tokenizer (Tokenizer): Instance of the Tokenizer class.
            transform (A.Compose, optional): Albumentations transforms.
            mode (str): 'train', 'valid', or 'test'. Determines the output of __getitem__.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.mode = mode

        # Pre-calculate full paths to avoid overhead in __getitem__
        # The metadata file_path is relative to input dir
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].values
        ]

        # For train/valid, pre-tokenize targets if possible, or handle in getitem
        # Given the dataset size, pre-tokenizing text is memory efficient enough and faster
        self.labels = None
        if self.mode in ["train", "valid"]:
            self.inchi_texts = df["InChI"].values
            # We don't pre-convert to tensors here to save some setup time,
            # but we could. Doing it in getitem is standard.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)

        if image is None:
            # Fallback for missing images (though validation showed 0 missing)
            # Create a black image
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple ToTensor if no transform provided
            image = ToTensorV2()(image=image)["image"]

        # 3. Return Data based on mode
        if self.mode in ["train", "valid"]:
            text = self.inchi_texts[idx]
            sequence = self.tokenizer.text_to_sequence(text)
            seq_len = len(sequence)

            # Convert to tensor
            label = torch.LongTensor(sequence)

            return image, label, seq_len

        else:  # test mode
            image_id = self.df.iloc[idx]["image_id"]
            return image, image_id
