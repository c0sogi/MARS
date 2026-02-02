import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# Import provided utilities and tokenizer class
from library.utils import get_atom_counts, ATOM_LIST
from library.tokenizer import Tokenizer


def get_transforms(img_size=256):
    """
    Returns the image transformations.
    Resizes images to a fixed size and applies ImageNet normalization.
    No heavy augmentation is used to ensure a stable baseline.

    Args:
        img_size (int): Target height and width of the image.

    Returns:
        A.Compose: Composed albumentations transforms.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


class ChemicalDataset(Dataset):
    """
    PyTorch Dataset for loading chemical images and InChI labels.
    Supports Train, Validation, and Test modes.
    Integrates auxiliary atom counting task.
    """

    def __init__(
        self,
        metadata_path,
        tokenizer,
        transform=None,
        mode="train",
        load_cached_data=True,
        cache_dir="./working/idea_6/",
        debug_size=None,
        input_root="./input",
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            tokenizer (Tokenizer): Instance of the Tokenizer class.
            transform (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached auxiliary data.
            cache_dir (str): Directory to save/load cached data.
            debug_size (int): If provided, limits the dataset size for debugging.
            input_root (str): Root directory of the input data.
        """
        self.mode = mode
        self.transform = transform
        self.tokenizer = tokenizer
        self.input_root = input_root
        self.cache_dir = cache_dir

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Debugging: subsample the dataset if debug_size is set
        if debug_size is not None and debug_size > 0:
            # Ensure we don't try to sample more than available
            limit = min(debug_size, len(self.df))
            self.df = self.df.iloc[:limit].reset_index(drop=True)
            print(f"[{mode.upper()}] Debug mode: subsampled to {len(self.df)} samples.")

        # Handle auxiliary data for training/validation
        self.atom_counts = None
        if self.mode in ["train", "val"]:
            # Ensure 'InChI' column exists
            if "InChI" not in self.df.columns:
                raise ValueError(
                    "Metadata must contain 'InChI' column for train/val modes."
                )

            # Get atom counts using the provided utility function (handles caching)
            # The utility function computes counts for the entire dataframe passed to it.
            # Since we might have subsampled self.df, the cache will be specific to this subset
            # (get_atom_counts uses len(df) in filename to distinguish).
            self.atom_counts = get_atom_counts(
                self.df, load_cached_data=load_cached_data, cache_dir=self.cache_dir
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        file_path = os.path.join(self.input_root, row["file_path"])

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            # Robustness: return a black image if file read fails
            # This prevents the entire training loop from crashing due to one bad file
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided: just convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.mode in ["train", "val"]:
            inchi_text = row["InChI"]

            # Tokenize text
            # text_to_sequence adds <SOS> and <EOS>
            sequence_indices = self.tokenizer.text_to_sequence(inchi_text)
            sequence_tensor = torch.tensor(sequence_indices, dtype=torch.long)

            # Retrieve pre-calculated atom counts
            atom_counts_vec = self.atom_counts[idx]
            atom_counts_tensor = torch.tensor(atom_counts_vec, dtype=torch.float32)

            return {
                "image": image,
                "sequence": sequence_tensor,
                "atom_counts": atom_counts_tensor,
                "original_text": inchi_text,
            }
        else:
            # Test mode: return image and ID for submission
            image_id = row["image_id"]
            return {"image": image, "image_id": image_id}
