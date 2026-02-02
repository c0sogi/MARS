import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config
from library.utils import seed_everything


def get_transforms(data_split: str):
    """
    Returns the appropriate transformation pipeline based on the data split.

    Args:
        data_split (str): One of 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if data_split == "train":
        # Training pipeline: Resize -> RandAugment -> Tensor -> Normalize
        return transforms.Compose(
            [
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                transforms.RandAugment(
                    num_ops=Config.RANDAUGMENT_NUM_OPS,
                    magnitude=Config.RANDAUGMENT_MAGNITUDE,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif data_split in ["val", "test"]:
        # Validation/Test pipeline: Resize -> Tensor -> Normalize
        return transforms.Compose(
            [
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        raise ValueError(f"Unknown data_split: {data_split}")


class CassavaDataset(Dataset):
    """
    PyTorch Dataset for Cassava Leaf Disease Classification.
    Handles loading images from disk, applying transforms, and caching metadata.
    """

    def __init__(
        self,
        metadata_path,
        transform=None,
        load_cached_data=True,
        data_split="train",
        subset_size=None,
    ):
        """
        Args:
            metadata_path (str): Path to the CSV file containing metadata.
            transform (callable, optional): Optional transform to be applied on a sample.
            load_cached_data (bool): Whether to try loading cached metadata.
            data_split (str): The subset of data ('train', 'val', 'test').
            subset_size (int, optional): If provided, limits the dataset to this many samples (for debugging).
        """
        self.transform = transform
        self.data_split = data_split
        self.subset_size = subset_size
        self.input_dir = Config.INPUT_DIR

        # Ensure working directory exists for cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Define cache file path
        cache_filename = f"cached_{data_split}_metadata.parquet"
        if subset_size is not None:
            cache_filename = (
                f"cached_{data_split}_subset_{subset_size}_metadata.parquet"
            )

        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        df = None

        # 1. IF load_cached_data is True: Try to load the file.
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
            except Exception:
                # If loading fails, proceed to load from source
                df = None

        # 2. IF loading fails OR load_cached_data is False: Compute/Process and Save.
        if df is None:
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

            df = pd.read_csv(metadata_path)

            # Apply subsetting if requested
            if self.subset_size is not None:
                df = df.iloc[: self.subset_size]

            # Save to cache
            # We use parquet as requested (no pickle)
            df.to_parquet(cache_path, index=False)

        # Store data in memory
        self.image_ids = df["image_id"].values
        self.labels = df["label"].values
        self.file_paths = df["file_path"].values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Construct full image path
        # metadata 'file_path' is relative to input dir (e.g. "train_images/xyz.jpg")
        img_path = os.path.join(self.input_dir, self.file_paths[idx])

        # Load image
        try:
            # Use PIL as per strategy
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Fallback for robustness (though data verification passed)
            # Return a black image of correct size
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (0, 0, 0))

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Return image and label (standard PyTorch format)
        # image_id can be accessed via self.image_ids[idx] if needed by the caller
        return image, label
