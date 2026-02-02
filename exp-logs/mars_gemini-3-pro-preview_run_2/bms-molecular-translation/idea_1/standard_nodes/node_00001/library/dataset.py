import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.tokenizer import Tokenizer


def load_metadata(
    csv_path: str, cache_path: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads metadata from CSV or Parquet cache following the required logic.
    """
    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If loading fails, proceed to load from scratch
            pass

    # 2. IF loading fails OR load_cached_data is False:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Save the result to the cache directory
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    # 3. Return the data.
    return df


class ChemicalDataset(Dataset):
    def __init__(
        self, mode: str = "train", transform=None, load_cached_data: bool = True
    ):
        """
        PyTorch Dataset for InChI chemical structure recognition.

        Args:
            mode (str): One of 'train', 'val', 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            load_cached_data (bool): Whether to attempt loading cached metadata (parquet).
        """
        self.mode = mode
        self.transform = transform
        self.tokenizer = Tokenizer()

        # Determine paths based on mode
        if self.mode == "train":
            csv_path = Config.TRAIN_CSV
            cache_filename = "train_metadata.parquet"
        elif self.mode == "val":
            csv_path = Config.VAL_CSV
            cache_filename = "val_metadata.parquet"
        elif self.mode == "test":
            csv_path = Config.TEST_CSV
            cache_filename = "test_metadata.parquet"
        else:
            raise ValueError(
                f"Unknown mode: {self.mode}. Must be 'train', 'val', or 'test'."
            )

        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # Load metadata
        self.df = load_metadata(csv_path, cache_path, load_cached_data)

        # Handle Debugging
        if Config.DEBUG:
            if len(self.df) > Config.DEBUG_SAMPLE_SIZE:
                self.df = self.df.sample(
                    n=Config.DEBUG_SAMPLE_SIZE, random_state=42
                ).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # --- Image Loading & Preprocessing ---
        file_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, file_path)

        # Load image in grayscale
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        # Handle cases where image might fail to load (robustness)
        if image is None:
            # Create a blank white image of default size
            image = (
                np.ones((Config.IMAGE_HEIGHT, Config.IMAGE_HEIGHT), dtype=np.uint8)
                * 255
            )

        h, w = image.shape
        target_h = Config.IMAGE_HEIGHT
        target_w = Config.IMAGE_WIDTH

        # Resize preserving aspect ratio
        scale = target_h / h
        new_w = int(w * scale)

        # Resize image
        if new_w > target_w:
            # If wider than target, squash to target width
            image = cv2.resize(image, (target_w, target_h))
            new_w = target_w
        else:
            image = cv2.resize(image, (new_w, target_h))

        # Pad to fixed width with white background (255)
        canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
        canvas[:, :new_w] = image

        # Normalize to [0, 1] and Invert
        # Text is black (0) on white (255). We want text to be high signal (1.0).
        image_norm = 1.0 - (canvas.astype(np.float32) / 255.0)

        # Convert to Tensor (C, H, W)
        image_tensor = torch.from_numpy(image_norm).unsqueeze(0)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        # --- Label Handling ---
        if self.mode in ["train", "val"]:
            inchi_text = row["InChI"]
            # Convert text to sequence of indices
            label_seq = self.tokenizer.text_to_sequence(inchi_text)
            # Length is needed for CTC Loss
            label_len = torch.tensor(len(label_seq), dtype=torch.long)

            return image_tensor, label_seq, label_len

        else:
            # Test mode: return image_id for submission mapping
            image_id = row["image_id"]
            return image_tensor, image_id
