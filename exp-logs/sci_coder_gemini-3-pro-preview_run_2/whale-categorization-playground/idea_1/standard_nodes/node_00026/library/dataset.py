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


class WhaleClassificationDataset(Dataset):
    """
    Dataset for training a classification model (ArcFace).
    Yields (image, label_idx).
    """

    def __init__(
        self,
        df,
        label_encoder,
        load_cached_data=True,
        transform=None,
        cache_name="train_images_cache.npy",
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            label_encoder (dict): Mapping from Id string to integer.
            load_cached_data (bool): Whether to use cached image data.
            transform (callable, optional): Transform to be applied on a sample.
            cache_name (str): Filename for the image cache.
        """
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.transform = transform or get_transforms("train")

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
        row = self.df.iloc[index]
        whale_id = row["Id"]
        filename = row["Image"]

        label_idx = self.label_encoder[whale_id]
        img_array = self.image_cache[filename]

        if self.transform:
            img = self.transform(img_array)

        return img, torch.tensor(label_idx, dtype=torch.long)


class SiameseWhaleDataset(Dataset):
    """
    Dataset for training a Siamese Network.
    Yields ((img1, img2), label), where label is 1.0 for same class, 0.0 for different.
    """

    def __init__(
        self,
        df,
        load_cached_data=True,
        transform=None,
        cache_name="train_pairs_cache.npy",
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

        # Pre-load images into RAM
        self.image_cache = cache_images(
            self.df, cache_name, load_cached_data=load_cached_data
        )

        # Filter dataframe to only include images that were successfully loaded
        available_images = set(self.image_cache.keys())
        self.df = self.df[self.df["Image"].isin(available_images)].reset_index(
            drop=True
        )

        # Group images by Id for efficient sampling
        self.group_examples = self.df.groupby("Id")["Image"].apply(list).to_dict()
        self.unique_ids = list(self.group_examples.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Anchor
        row = self.df.iloc[index]
        anchor_id = row["Id"]
        anchor_img_name = row["Image"]

        # Determine if positive or negative pair (50/50 split)
        # If class has only 1 image (singleton), we must generate a negative pair
        is_singleton = len(self.group_examples[anchor_id]) < 2
        should_get_positive = (random.random() > 0.5) and not is_singleton

        if should_get_positive:
            target_label = 1.0
            # Positive: Pick another image from the same class
            candidates = self.group_examples[anchor_id]
            target_img_name = random.choice(candidates)
            # Try to pick a different image, but if only 2 exist, it's the other one.
            # If we pick the same image, distance is 0, which is valid (loss=0).
            # We try a few times to get a different one.
            for _ in range(3):
                if target_img_name != anchor_img_name:
                    break
                target_img_name = random.choice(candidates)
        else:
            target_label = 0.0
            # Negative: Pick an image from a different class
            target_id = random.choice(self.unique_ids)
            while target_id == anchor_id:
                target_id = random.choice(self.unique_ids)
            target_img_name = random.choice(self.group_examples[target_id])

        img1_array = self.image_cache[anchor_img_name]
        img2_array = self.image_cache[target_img_name]

        if self.transform:
            img1 = self.transform(img1_array)
            img2 = self.transform(img2_array)

        return (img1, img2), torch.tensor(target_label, dtype=torch.float)


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
