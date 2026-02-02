import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.tokenizer import InChITokenizer


def get_transforms(phase: str):
    """
    Returns the albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    if phase == "train":
        return A.Compose(
            [
                # Light rotation to simulate scan misalignment
                A.SafeRotate(
                    limit=5, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255
                ),
                # Noise injection to simulate scan quality issues
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                    ],
                    p=0.5,
                ),
                # Normalize using stats from EDA (Grayscale)
                # Mean ~0.98 (white background), Std ~0.15
                A.Normalize(mean=(0.98,), std=(0.15,), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Only normalization and tensor conversion
        return A.Compose(
            [
                A.Normalize(mean=(0.98,), std=(0.15,), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )


class InChIDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        tokenizer: InChITokenizer,
        transform=None,
        phase: str = "train",
    ):
        """
        Dataset class for InChI chemical structure images.

        Args:
            metadata_path (str): Path to the metadata CSV file (train.csv, val.csv, test.csv).
            tokenizer (InChITokenizer): Instance of the tokenizer for label encoding.
            transform (albumentations.Compose, optional): Augmentation pipeline.
            phase (str): Current phase ('train', 'val', 'test'). Used to determine return format.
        """
        self.df = pd.read_csv(metadata_path)
        self.tokenizer = tokenizer
        self.transform = transform
        self.phase = phase
        self.input_dir = Config.INPUT_DIR
        self.img_h = Config.IMAGE_HEIGHT
        self.max_w = Config.MAX_WIDTH

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Load image in grayscale (1 channel)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Robustness check
        if image is None:
            # Create a blank white image if load fails
            image = np.full((self.img_h, self.img_h), 255, dtype=np.uint8)

        # Resize to fixed height, maintaining aspect ratio
        h, w = image.shape
        scale = self.img_h / h
        new_w = int(w * scale)

        # Clip extremely wide images to prevent OOM
        if new_w > self.max_w:
            new_w = self.max_w

        image = cv2.resize(image, (new_w, self.img_h))

        # Expand dims for albumentations: (H, W) -> (H, W, 1)
        image = np.expand_dims(image, axis=2)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback manual conversion if no transform provided
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Return data based on phase
        if self.phase in ["train", "val"]:
            inchi_text = row["InChI"]
            # Tokenize text
            seq = self.tokenizer.text_to_sequence(inchi_text)
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            # Return image and token sequence
            return image, seq_tensor
        else:
            # Test phase: return image and ID for submission
            image_id = row["image_id"]
            return image, image_id


def collate_fn(batch):
    """
    Custom collate function for dynamic padding of variable-width images and sequences.

    Args:
        batch: List of tuples returned by InChIDataset.__getitem__
               Train/Val: [(image, seq_tensor), ...]
               Test:      [(image, image_id), ...]

    Returns:
        Batched tensors appropriately padded.
    """
    images, targets = zip(*batch)

    batch_size = len(images)
    c, h, _ = images[0].shape

    # ----------------------------------------------------------------
    # 1. Dynamic Image Padding
    # ----------------------------------------------------------------
    # Find the maximum width in this batch
    widths = [img.shape[2] for img in images]
    max_w = max(widths)

    # Round up max_w to the nearest multiple of PAD_MULTIPLE (e.g., 32)
    # This ensures the dimensions are compatible with ResNet downsampling
    pad_multiple = Config.PAD_MULTIPLE
    padded_w = int(np.ceil(max_w / pad_multiple) * pad_multiple)

    # Create batch tensor initialized with 0.0
    # Note: With our normalization (mean=0.98), 0.0 in the tensor corresponds to
    # a pixel value of roughly 250 (near white/background).
    # So 0-padding effectively pads with background color.
    batch_images = torch.zeros(batch_size, c, h, padded_w)

    # Place images into the batch tensor (left-aligned)
    for i, img in enumerate(images):
        w = img.shape[2]
        batch_images[i, :, :, :w] = img

    # ----------------------------------------------------------------
    # 2. Target Handling
    # ----------------------------------------------------------------
    # Check if targets are tensors (train/val labels) or strings (test ids)
    if isinstance(targets[0], torch.Tensor):
        # Train/Val phase: Pad sequences
        lengths = [len(t) for t in targets]
        max_len = max(lengths)

        # Pad with tokenizer.PAD_IDX (which is 0)
        pad_idx = 0

        batch_targets = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)

        for i, seq in enumerate(targets):
            end = lengths[i]
            batch_targets[i, :end] = seq

        # Return images, padded targets, and original lengths
        return batch_images, batch_targets, torch.tensor(lengths)

    else:
        # Test phase: targets are image_ids
        return batch_images, list(targets)
