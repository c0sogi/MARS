import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import seed_everything

# Constants
CACHE_DIR = "./working/idea_16/"
INPUT_DIR = "./input"


class BirdDataset(Dataset):
    def __init__(
        self, df, image_dict, transforms=None, phase="train", time_roll_prob=0.0
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata (rec_id, file_paths, labels).
            image_dict (dict): Dictionary mapping rec_id to numpy image arrays.
            transforms (A.Compose): Albumentations pipeline.
            phase (str): 'train', 'val', or 'test'.
            time_roll_prob (float): Probability of applying TimeRoll augmentation.
        """
        self.df = df.reset_index(drop=True)
        self.image_dict = image_dict
        self.transforms = transforms
        self.phase = phase
        self.time_roll_prob = time_roll_prob

        # Identify label columns (species_0 to species_18)
        self.label_cols = [c for c in df.columns if c.startswith("species_")]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = int(row["rec_id"])

        # Retrieve image (H, W) - Grayscale
        # Use copy to avoid modifying the cached version in memory
        if rec_id in self.image_dict:
            img = self.image_dict[rec_id].copy()
        else:
            # Fallback: create black image (should not happen with proper caching)
            img = np.zeros((256, 1246), dtype=np.uint8)

        # 1. Time Roll (Circular Shift)
        # Applied on raw spectrogram along time axis (width, axis=1)
        # This augmentation is crucial for learning translation invariance of bird calls
        if self.phase == "train" and np.random.rand() < self.time_roll_prob:
            shift = np.random.randint(0, img.shape[1])
            img = np.roll(img, shift, axis=1)

        # 2. Pseudo-RGB (Channel Replication)
        # Convert (H, W) -> (H, W, 3) to match ImageNet pretrained weights
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 3. Albumentations (Resize, Augment, Normalize, ToTensor)
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]

        # 4. Labels
        # For test set, these will be placeholders (0s), which is handled by the metadata generation
        labels = row[self.label_cols].values.astype(np.float32)

        return {"id": rec_id, "image": img, "targets": torch.tensor(labels)}


def load_and_cache_images(df, cache_dir, load_cached_data):
    """
    Loads images from disk or cache.
    Cache format: images.npy (N, H, W) and rec_ids.npy (N,)
    """
    os.makedirs(cache_dir, exist_ok=True)
    images_path = os.path.join(cache_dir, "images.npy")
    ids_path = os.path.join(cache_dir, "rec_ids.npy")

    image_dict = {}
    cache_loaded = False

    # Try loading from cache
    if load_cached_data and os.path.exists(images_path) and os.path.exists(ids_path):
        try:
            images_arr = np.load(images_path)
            ids_arr = np.load(ids_path)
            # Reconstruct dictionary
            for i, rec_id in enumerate(ids_arr):
                image_dict[rec_id] = images_arr[i]
            cache_loaded = True
        except Exception as e:
            print(f"Cache load failed: {e}. Recomputing...")

    # Check if we have all needed images for the current dataframe
    needed_ids = df["rec_id"].unique()
    missing_ids = [rid for rid in needed_ids if rid not in image_dict]

    if len(missing_ids) > 0:
        # Load missing images from disk
        subset_df = df[df["rec_id"].isin(missing_ids)].drop_duplicates(
            subset=["rec_id"]
        )

        for _, row in subset_df.iterrows():
            rec_id = int(row["rec_id"])
            rel_path = row["file_path_spec"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            if os.path.exists(full_path):
                # Load as grayscale
                img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    image_dict[rec_id] = img

        # Save cache if we performed a fresh load or forced rebuild
        # We save all currently loaded images to the cache
        if not cache_loaded or not load_cached_data:
            if len(image_dict) > 0:
                ids_list = np.array(list(image_dict.keys()))
                # Stack images (assuming same size, which EDA confirmed: 256x1246)
                imgs_list = np.array([image_dict[i] for i in ids_list])

                np.save(images_path, imgs_list)
                np.save(ids_path, ids_list)

    return image_dict


def get_loader(
    df,
    model_name,
    phase="train",
    batch_size=32,
    num_workers=2,
    load_cached_data=True,
    cache_dir=CACHE_DIR,
):
    """
    Creates a DataLoader for the BirdDataset.

    Args:
        df (pd.DataFrame): Dataframe with samples.
        model_name (str): Name of the model architecture (determines resolution).
        phase (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        num_workers (int): Number of DataLoader workers.
        load_cached_data (bool): Whether to use cached images.
        cache_dir (str): Directory for cache files.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    # Determine Resolution based on architecture
    # DenseNet uses lower resolution to focus on macro features
    if "densenet" in model_name.lower():
        height, width = 160, 320
    else:
        # ResNet18 and EfficientNet-B0 use standard/higher resolution
        height, width = 224, 448

    # Define Transforms
    if phase == "train":
        transform = A.Compose(
            [
                A.Resize(height, width),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        # Always apply random roll during training to maximize data utility
        time_roll_prob = 1.0
    else:
        transform = A.Compose(
            [
                A.Resize(height, width),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        time_roll_prob = 0.0

    # Load Data (with Caching)
    image_dict = load_and_cache_images(df, cache_dir, load_cached_data)

    # Create Dataset
    dataset = BirdDataset(
        df=df,
        image_dict=image_dict,
        transforms=transform,
        phase=phase,
        time_roll_prob=time_roll_prob,
    )

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(phase == "train"),
        worker_init_fn=lambda id: seed_everything(42 + id),
    )

    return loader
