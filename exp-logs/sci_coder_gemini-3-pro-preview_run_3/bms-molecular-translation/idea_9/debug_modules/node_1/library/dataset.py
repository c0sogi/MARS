import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.utils import compute_and_cache_atom_counts, ATOM_VOCAB
from library.tokenizer import InchiTokenizer

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_9/"
IMG_SIZE = 384


def get_transforms(phase: str):
    """
    Returns the image transformations for the specified phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Albumentations transformation pipeline.
    """
    # Base transforms: Resize and Normalize
    # We use standard ImageNet normalization statistics as a robust default
    # for pre-trained backbones (like the MLP-Mixer/ResNet mentioned in ideas).
    transforms_list = [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]

    return A.Compose(transforms_list)


class InchiDataset(Dataset):
    """
    PyTorch Dataset for InChI chemical structure recognition.
    Handles image loading, resizing, tokenization, and auxiliary target generation.
    """

    def __init__(
        self,
        metadata_path: str,
        tokenizer: InchiTokenizer,
        max_length: int = 400,
        mode: str = "train",
        load_cached_data: bool = True,
        transform: A.Compose = None,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            tokenizer (InchiTokenizer): Instance of the tokenizer.
            max_length (int): Maximum sequence length for tokenization.
            mode (str): 'train', 'valid', or 'test'.
            load_cached_data (bool): Whether to use cached auxiliary data.
            transform (A.Compose): Albumentations transforms.
        """
        self.mode = mode
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.transform = transform if transform is not None else get_transforms(mode)

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Handle auxiliary targets (Atom Counts) for training/validation
        self.atom_counts = None
        if self.mode in ["train", "valid"]:
            # Ensure cache directory exists
            os.makedirs(CACHE_DIR, exist_ok=True)

            # Use the provided utility function which implements the caching logic
            # logic: check cache -> load if exists and allowed -> else compute -> save
            self.atom_counts = compute_and_cache_atom_counts(
                metadata_path=metadata_path, load_cached_data=load_cached_data
            )

            # Sanity check
            if len(self.atom_counts) != len(self.df):
                raise ValueError(
                    f"Mismatch between metadata rows ({len(self.df)}) "
                    f"and atom counts ({len(self.atom_counts)})"
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        file_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, file_path)

        # Read image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (should ideally not happen given validation)
            # Create a black image
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            # Fallback to simple tensor conversion
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # 3. Prepare Targets
        if self.mode in ["train", "valid"]:
            inchi_text = row["InChI"]

            # Tokenize
            # padding=True ensures we get a consistent tensor size if batching manually,
            # though usually collate_fn handles padding. Here we pad to max_length for simplicity.
            token_ids = self.tokenizer.encode(
                inchi_text, max_length=self.max_length, padding=True
            )

            # Get sequence length (useful for masking in Transformer)
            # Count non-pad tokens
            seq_len = torch.sum(token_ids != self.tokenizer.pad_token_id).item()

            # Get auxiliary atom counts
            atom_vec = torch.tensor(self.atom_counts[idx], dtype=torch.float32)

            return {
                "image": image_tensor,
                "input_ids": token_ids,
                "seq_len": seq_len,
                "atom_counts": atom_vec,
                "inchi_text": inchi_text,  # Useful for debugging/logging
            }

        else:
            # Test mode
            image_id = row["image_id"]
            return {"image": image_tensor, "image_id": image_id}
