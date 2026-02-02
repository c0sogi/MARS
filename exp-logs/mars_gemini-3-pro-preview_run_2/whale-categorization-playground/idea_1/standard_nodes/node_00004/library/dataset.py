import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the torchvision transformations for training or inference.

    Args:
        mode (str): 'train' or 'val'/'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=Config.MEAN, std=Config.STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=Config.MEAN, std=Config.STD),
            ]
        )


def cache_images(df, cache_name, load_cached_data=True):
    """
    Loads images into memory and caches them to disk as a numpy dictionary.

    Args:
        df (pd.DataFrame): DataFrame containing 'file_path' and 'Image' columns.
        cache_name (str): Name of the cache file (e.g., 'train_images.npy').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary mapping image filename to numpy array (RGB).
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading images from cache: {cache_path}")
            # Allow_pickle=True is required for saving/loading dicts with numpy
            data = np.load(cache_path, allow_pickle=True).item()
            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing images for {cache_name}...")
    image_dict = {}

    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        filename = row["Image"]

        if os.path.exists(full_path):
            # Read with OpenCV
            img = cv2.imread(full_path)
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize here to save RAM, though transforms also resize
                # Keeping it slightly larger or exact size is fine.
                # We resize to exact size to optimize storage.
                img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
                image_dict[filename] = img

    # 3. Save to cache
    print(f"Saving {len(image_dict)} images to cache: {cache_path}")
    np.save(cache_path, image_dict)

    return image_dict


class SiameseWhaleDataset(Dataset):
    """
    Dataset for training a Siamese Network.
    Yields pairs of images (anchor, other) and a binary label (same/different).
    """

    def __init__(
        self,
        df,
        load_cached_data=True,
        transform=None,
        cache_name="train_images_cache.npy",
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            load_cached_data (bool): Whether to use cached image data.
            transform (callable, optional): Transform to be applied on a sample.
            cache_name (str): Filename for the image cache.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform or get_transforms("train")

        # Group images by Id for fast sampling
        self.id_to_images = self.df.groupby("Id")["Image"].apply(list).to_dict()
        self.ids = list(self.id_to_images.keys())

        # Pre-load images into RAM
        self.image_cache = cache_images(
            self.df, cache_name, load_cached_data=load_cached_data
        )

        # Filter dataframe to only include images that were successfully loaded
        available_images = set(self.image_cache.keys())
        self.df = self.df[self.df["Image"].isin(available_images)].reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Anchor
        row = self.df.iloc[index]
        anchor_id = row["Id"]
        anchor_filename = row["Image"]

        # Determine if positive or negative pair (50/50)
        # We use random.random() < 0.5.
        # Note: We use numpy random for consistency if seeded externally
        is_positive = np.random.random() < 0.5

        if is_positive:
            label = 1.0
            # Get all images for this whale ID
            candidates = self.id_to_images[anchor_id]

            # Filter candidates to those available in cache
            candidates = [c for c in candidates if c in self.image_cache]

            if len(candidates) > 1:
                # Pick a different image if possible, or same (rarely)
                # To ensure robustness, we can pick any from the list
                second_filename = np.random.choice(candidates)
            else:
                # Singleton: Use the same image.
                # The transform will create a synthetic view.
                second_filename = anchor_filename
        else:
            label = 0.0
            # Pick a different class
            while True:
                neg_id = np.random.choice(self.ids)
                if neg_id != anchor_id:
                    break

            # Pick a random image from that class
            neg_candidates = self.id_to_images[neg_id]
            # Filter
            neg_candidates = [c for c in neg_candidates if c in self.image_cache]

            if not neg_candidates:
                # Fallback if the chosen negative class has no valid images (rare edge case)
                # Just use anchor as negative (label 0) - creates noise but prevents crash
                second_filename = anchor_filename
                label = 1.0  # Revert to positive if we can't find negative
            else:
                second_filename = np.random.choice(neg_candidates)

        # Retrieve images from cache
        img1_array = self.image_cache[anchor_filename]
        img2_array = self.image_cache[second_filename]

        # Apply transforms
        # Transforms expect PIL Image or Tensor. ToPILImage handles ndarray (H,W,C)
        if self.transform:
            img1 = self.transform(img1_array)
            img2 = self.transform(img2_array)

        return (img1, img2), torch.tensor(label, dtype=torch.float32)


class WhaleInferenceDataset(Dataset):
    """
    Dataset for inference/validation. Yields single images and their Id.
    """

    def __init__(
        self, df, load_cached_data=True, transform=None, cache_name="val_test_cache.npy"
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            load_cached_data (bool): Whether to use cached image data.
            transform (callable, optional): Transform pipeline.
            cache_name (str): Unique name for the cache file.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform or get_transforms("val")

        # Pre-load images
        self.image_cache = cache_images(
            self.df, cache_name, load_cached_data=load_cached_data
        )

        # Filter
        available_images = set(self.image_cache.keys())
        self.df = self.df[self.df["Image"].isin(available_images)].reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        filename = row["Image"]
        whale_id = row["Id"] if "Id" in row and pd.notna(row["Id"]) else "unknown"

        img_array = self.image_cache[filename]

        if self.transform:
            img = self.transform(img_array)

        return img, whale_id, filename
